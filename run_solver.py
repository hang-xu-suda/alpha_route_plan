import sys
import os
import pickle
import gzip
from datetime import datetime
import numpy as np
import time as time_module
from collections import defaultdict
from typing import Dict, Tuple
from pathlib import Path  # ← 添加这行
# ════════════════════════════════════════════════════════════════
# 全局变量：数据（只加载一次）
# ════════════════════════════════════════════════════════════════

# 原有全局数据变量
G_GLOBAL = None
SPARSE_DATA_GLOBAL = None
NODE_TO_INDEX_GLOBAL = None
SCENARIO_DATES_GLOBAL = None
SCENARIO_PROBS_GLOBAL = None
TIME_INTERVALS_PER_DAY_GLOBAL = None

# ✨ 新增：预计算的数据结构
ADJ_LIST_FORWARD_GLOBAL = None  # 正向邻接表
ADJ_LIST_BACKWARD_GLOBAL = None  # 反向邻接表
LINK_DISTRIBUTIONS_GLOBAL = None  # 链路分布
LINK_DISTRIBUTIONS_BACKWARD_GLOBAL = None
EDGE_TRAVEL_TIME_BOUNDS_GLOBAL=None
DATA_LOADED = False

def load_data_once(data_path=None, cache_file='precomputed_data.pkl.gz', force_rebuild=False):
    """
    全局加载数据（只加载一次）
    ✨ 新增：同时预计算邻接表和链路分布，支持缓存
    
    Args:
        data_path: 数据文件路径
        cache_file: 缓存文件路径
        force_rebuild: 是否强制重建缓存（忽略已有缓存）
    """
    global G_GLOBAL, SPARSE_DATA_GLOBAL, NODE_TO_INDEX_GLOBAL, INDEX_TO_NODE_GLOBAL
    global SCENARIO_DATES_GLOBAL, SCENARIO_PROBS_GLOBAL, TIME_INTERVALS_PER_DAY_GLOBAL
    global ADJ_LIST_FORWARD_GLOBAL, ADJ_LIST_BACKWARD_GLOBAL, LINK_DISTRIBUTIONS_GLOBAL
    global LINK_DISTRIBUTIONS_BACKWARD_GLOBAL,EDGE_TRAVEL_TIME_BOUNDS_GLOBAL  # ← 新增：用于反向求解
    global DATA_LOADED
    
    if DATA_LOADED and not force_rebuild:
        print("数据已加载，跳过重复加载")
        return (G_GLOBAL, SPARSE_DATA_GLOBAL, NODE_TO_INDEX_GLOBAL,
                SCENARIO_DATES_GLOBAL, SCENARIO_PROBS_GLOBAL, TIME_INTERVALS_PER_DAY_GLOBAL)
    
    # 导入config
    try:
        import config as config
        if data_path is None: 
            data_path = config.DATA_PATH
    except: 
        if data_path is None:
            data_path = 'data/test_data.pkl.gz'
    
    print(f"\n{'='*70}")
    print(f"加载和预处理数据（优化版 - 支持缓存）")
    print(f"{'='*70}")
    print(f"  数据文件: {data_path}")
    print(f"  缓存文件: {cache_file}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    start_time = time_module.time()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤1：加载基础数据
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n[1/4] 加载基础数据...")
    with gzip.open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    G_GLOBAL = data['G']
    SPARSE_DATA_GLOBAL = data['sparse_data']
    NODE_TO_INDEX_GLOBAL = data['node_to_index']
    INDEX_TO_NODE_GLOBAL = {v: k for k, v in NODE_TO_INDEX_GLOBAL.items()}
    SCENARIO_DATES_GLOBAL = [datetime.strptime(d, '%Y-%m-%d').date() 
                              for d in data['scenario_dates']]
    SCENARIO_PROBS_GLOBAL = data['scenario_probs']
    TIME_INTERVALS_PER_DAY_GLOBAL = data['time_intervals_per_day']
    
    print(f"  ✓ 基础数据加载完成")
    print(f"    节点数: {len(G_GLOBAL.nodes()):,}")
    print(f"    边数: {len(G_GLOBAL.edges()):,}")
    print(f"    场景数: {len(SCENARIO_DATES_GLOBAL)}")
    print(f"    时间片数/天: {TIME_INTERVALS_PER_DAY_GLOBAL}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤2：尝试从缓存加载预计算数据
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    cache_loaded = False
    
    if not force_rebuild and Path(cache_file).exists():
        print(f"\n[2/4] 尝试从缓存加载预计算数据...")
        cache_result = load_precomputed_data(cache_file)
        
        if cache_result is not None:
            ADJ_LIST_FORWARD_GLOBAL, ADJ_LIST_BACKWARD_GLOBAL, LINK_DISTRIBUTIONS_GLOBAL, LINK_DISTRIBUTIONS_BACKWARD_GLOBAL,EDGE_TRAVEL_TIME_BOUNDS_GLOBAL = cache_result
            cache_loaded = True
            print(f"  🚀 从缓存加载成功！")
        else:
            print(f"  ⚠️ 缓存加载失败，将重新计算")
    else:
        if force_rebuild:
            print(f"\n[2/4] 强制重建预计算数据...")
        else:
            print(f"\n[2/4] 缓存文件不存在，需要计算预计算数据...")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤3：如果缓存未加载，重新计算并保存
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if not cache_loaded:
        print(f"\n[3/4] 计算预计算数据...")
        
        # 3.1 构建邻接表
        print(f"  [3.1] 构建邻接表...")
        adj_start = time_module.time()
        
        ADJ_LIST_FORWARD_GLOBAL, ADJ_LIST_BACKWARD_GLOBAL = _build_adjacency_lists(
            SPARSE_DATA_GLOBAL,
            NODE_TO_INDEX_GLOBAL,
            len(SCENARIO_DATES_GLOBAL)
        )
        
        adj_time = time_module.time() - adj_start
        print(f"    ✓ 邻接表构建完成 (用时 {adj_time:.2f}秒)")
        print(f"      正向邻接表: {len(ADJ_LIST_FORWARD_GLOBAL)} 个节点")
        print(f"      反向邻接表:  {len(ADJ_LIST_BACKWARD_GLOBAL)} 个节点")
        
        # 3.2 预计算链路分布（同时生成正向和反向）
        print(f"  [3.2] 预计算链路分布...")
        dist_start = time_module.time()
        
        # 修改：接收两个返回值
        LINK_DISTRIBUTIONS_GLOBAL, LINK_DISTRIBUTIONS_BACKWARD_GLOBAL ,EDGE_TRAVEL_TIME_BOUNDS_GLOBAL= _precompute_link_distributions(
            SPARSE_DATA_GLOBAL,
            NODE_TO_INDEX_GLOBAL,
            len(SCENARIO_DATES_GLOBAL)
        )
        
        dist_time = time_module.time() - dist_start
        print(f"    ✓ 链路分布计算完成 (用时 {dist_time:.2f}秒)")
        
        # 3.3 保存到缓存
        print(f"\n  [3.3] 保存预计算数据到缓存...")
        save_precomputed_data(
            ADJ_LIST_FORWARD_GLOBAL,
            ADJ_LIST_BACKWARD_GLOBAL,
            LINK_DISTRIBUTIONS_GLOBAL,
            LINK_DISTRIBUTIONS_BACKWARD_GLOBAL,  # ← 新增参数
            EDGE_TRAVEL_TIME_BOUNDS_GLOBAL,
            filename=cache_file
        )
    else:
        # 如果从缓存加载，跳过步骤3
        print(f"\n[3/4] ✓ 跳过计算（已从缓存加载）")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤4：完成
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    load_time = time_module.time() - start_time
    
    print(f"\n[4/4] ✓ 数据加载和预处理完成！")
    print(f"  总耗时: {load_time:.2f}秒")
    if cache_loaded:
        print(f"    基础数据加载: {load_time:.2f}秒")
        print(f"    预计算数据:  从缓存加载 🚀")
    else:
        print(f"    基础数据加载: 约 {(load_time - adj_time - dist_time):.2f}秒")
        print(f"    邻接表构建: {adj_time:.2f}秒")
        print(f"    链路分布计算: {dist_time:.2f}秒")
        print(f"    缓存保存: 已完成 💾")
    print(f"{'='*70}\n")
    
    DATA_LOADED = True
    
    return (G_GLOBAL, SPARSE_DATA_GLOBAL, NODE_TO_INDEX_GLOBAL,
            SCENARIO_DATES_GLOBAL, SCENARIO_PROBS_GLOBAL, TIME_INTERVALS_PER_DAY_GLOBAL)


def save_precomputed_data(adj_list_forward, adj_list_backward, 
                          link_distributions_forward, link_distributions_backward,edge_travel_time_bounds,
                          filename='precomputed_data.pkl.gz'):
    """保存预计算数据（压缩，包含正向和反向链路分布）"""
    print(f"\n{'='*70}")
    print(f"保存预计算数据")
    print(f"{'='*70}")
    
    data = {
        'adj_list_forward': dict(adj_list_forward),
        'adj_list_backward':  dict(adj_list_backward),
        'link_distributions_forward': link_distributions_forward,  # ← 修改
        'link_distributions_backward':  link_distributions_backward,  # ← 新增
        'edge_travel_time_bounds':edge_travel_time_bounds,
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'forward_nodes': len(adj_list_forward),
            'reverse_nodes': len(adj_list_backward),
            'forward_edges': sum(len(v) for v in adj_list_forward.values()),
            'reverse_edges': sum(len(v) for v in adj_list_backward.values()),
            'distributions_forward': len(link_distributions_forward),  # ← 修改
            'distributions_backward':  len(link_distributions_backward)  # ← 新增
            
        }
    }
    
    start_time = time_module.time()
    
    # 保存为压缩文件
    with gzip.open(filename, 'wb', compresslevel=6) as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size = os.path.getsize(filename)
    elapsed = time_module.time() - start_time
    
    print(f"  ✓ 已保存:  {filename}")
    print(f"  文件大小: {file_size / 1024 / 1024:.2f} MB")
    print(f"  耗时: {elapsed:.2f}秒")
    print(f"{'='*70}\n")

def load_precomputed_data(filename='precomputed_data.pkl.gz'):
    """加载预计算数据（压缩，包含正向和反向链路分布）"""
    if not os.path.exists(filename):
        return None
    
    print(f"\n{'='*70}")
    print(f"加载预计算数据")
    print(f"{'='*70}")
    
    start_time = time_module.time()
    
    try:
        with gzip.open(filename, 'rb') as f:
            data = pickle.load(f)
        
        elapsed = time_module.time() - start_time
        
        metadata = data.get('metadata', {})
        
        print(f"  ✓ 已加载: {filename}")
        print(f"  数据时间:  {metadata.get('timestamp', 'unknown')}")
        print(f"  正向节点:  {metadata.get('forward_nodes', 0):,}")
        print(f"  反向节点: {metadata.get('reverse_nodes', 0):,}")
        print(f"  正向边数: {metadata.get('forward_edges', 0):,}")
        print(f"  反向边数: {metadata.get('reverse_edges', 0):,}")
        print(f"  正向链路分布: {metadata.get('distributions_forward', 0):,}")
        print(f"  反向链路分布: {metadata.get('distributions_backward', 0):,}")
        print(f"  加载耗时: {elapsed:.2f}秒")
        print(f"{'='*70}\n")
        
        # 转换回 defaultdict
        from collections import defaultdict
        adj_forward = defaultdict(list, data['adj_list_forward'])
        adj_backward = defaultdict(list, data['adj_list_backward'])
        edge_travel_time_bounds = defaultdict(list, data['edge_travel_time_bounds'])
        
        # 兼容旧版本缓存（只有一个link_distributions）
        if 'link_distributions_forward' in data:
            link_dists_forward = data['link_distributions_forward']
            link_dists_backward = data.get('link_distributions_backward', link_dists_forward)
        else:
            # 旧版本缓存
            print(f"  ⚠️ 检测到旧版本缓存，正向和反向使用相同分布")
            link_dists_forward = data.get('link_distributions', {})
            link_dists_backward = link_dists_forward
        
        return adj_forward, adj_backward, link_dists_forward, link_dists_backward,edge_travel_time_bounds
    
    except Exception as e: 
        print(f"  ✗ 加载失败:  {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}\n")
        return None


def get_precomputed_data():
    """
    ✨ 获取预计算的数据结构（从全局变量）
    
    Returns:
        tuple: (adj_list_forward, adj_list_backward, link_distributions_forward, link_distributions_backward)
    """
    global LINK_DISTRIBUTIONS_BACKWARD_GLOBAL  # 确保声明了这个全局变量
    
    if not DATA_LOADED:
        raise RuntimeError("数据未加载，请先调用 load_data_once()")
    
    # 兼容性：如果反向分布未定义，使用正向分布
    if LINK_DISTRIBUTIONS_BACKWARD_GLOBAL is None:
        print("  ⚠️ 反向链路分布未定义，使用正向分布")
        LINK_DISTRIBUTIONS_BACKWARD_GLOBAL = LINK_DISTRIBUTIONS_GLOBAL
    
    return (ADJ_LIST_FORWARD_GLOBAL, ADJ_LIST_BACKWARD_GLOBAL, 
            LINK_DISTRIBUTIONS_GLOBAL, LINK_DISTRIBUTIONS_BACKWARD_GLOBAL,EDGE_TRAVEL_TIME_BOUNDS_GLOBAL)


# def load_data_once(data_path=None):
#     """
#     全局加载数据（只加载一次）
#     ✨ 新增：同时预计算邻接表和链路分布
#     """
#     global G_GLOBAL, SPARSE_DATA_GLOBAL, NODE_TO_INDEX_GLOBAL
#     global SCENARIO_DATES_GLOBAL, SCENARIO_PROBS_GLOBAL, TIME_INTERVALS_PER_DAY_GLOBAL
#     global ADJ_LIST_FORWARD_GLOBAL, ADJ_LIST_BACKWARD_GLOBAL, LINK_DISTRIBUTIONS_GLOBAL
#     global DATA_LOADED
    
#     if DATA_LOADED:
#         print("数据已加载，跳过重复加载")
#         return (G_GLOBAL, SPARSE_DATA_GLOBAL, NODE_TO_INDEX_GLOBAL,
#                 SCENARIO_DATES_GLOBAL, SCENARIO_PROBS_GLOBAL, TIME_INTERVALS_PER_DAY_GLOBAL)
    
#     # 导入config
#     try:
#         import config as config
#         if data_path is None:
#             data_path = config.DATA_PATH
#     except: 
#         if data_path is None:
#             data_path = 'data/test_data.pkl.gz'
    
#     print(f"\n{'='*70}")
#     print(f"加载和预处理数据（优化版 - 只执行一次）")
#     print(f"{'='*70}")
#     print(f"  数据文件: {data_path}")
    
#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
#     start_time = time_module.time()
    
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     # 步骤1：加载基础数据
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     print(f"\n[1/4] 加载基础数据...")
#     with gzip.open(data_path, 'rb') as f:
#         data = pickle.load(f)
    
#     G_GLOBAL = data['G']
#     SPARSE_DATA_GLOBAL = data['sparse_data']
#     NODE_TO_INDEX_GLOBAL = data['node_to_index']
#     SCENARIO_DATES_GLOBAL = [datetime.strptime(d, '%Y-%m-%d').date() 
#                               for d in data['scenario_dates']]
#     SCENARIO_PROBS_GLOBAL = data['scenario_probs']
#     TIME_INTERVALS_PER_DAY_GLOBAL = data['time_intervals_per_day']
    
#     print(f"  ✓ 基础数据加载完成")
#     print(f"    节点数: {len(G_GLOBAL.nodes()):,}")
#     print(f"    边数: {len(G_GLOBAL.edges()):,}")
#     print(f"    场景数: {len(SCENARIO_DATES_GLOBAL)}")
#     print(f"    时间片数/天: {TIME_INTERVALS_PER_DAY_GLOBAL}")
    
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     # 步骤2：构建邻接表
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     print(f"\n[2/4] 构建邻接表...")
#     adj_start = time_module.time()
    
#     ADJ_LIST_FORWARD_GLOBAL, ADJ_LIST_BACKWARD_GLOBAL = _build_adjacency_lists(
#         SPARSE_DATA_GLOBAL,
#         NODE_TO_INDEX_GLOBAL,
#         len(SCENARIO_DATES_GLOBAL)
#     )
    
#     adj_time = time_module.time() - adj_start
#     print(f"  ✓ 邻接表构建完成 (用时 {adj_time:.2f}秒)")
#     print(f"    正向邻接表: {len(ADJ_LIST_FORWARD_GLOBAL)} 个节点")
#     print(f"    反向邻接表: {len(ADJ_LIST_BACKWARD_GLOBAL)} 个节点")
    
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     # 步骤3：预计算链路分布
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     print(f"\n[3/4] 预计算链路分布...")
#     dist_start = time_module.time()
    
#     LINK_DISTRIBUTIONS_GLOBAL = _precompute_link_distributions(
#         SPARSE_DATA_GLOBAL,
#         NODE_TO_INDEX_GLOBAL,
#         len(SCENARIO_DATES_GLOBAL)
#     )
    
#     dist_time = time_module.time() - dist_start
#     print(f"  ✓ 链路分布计算完成 (用时 {dist_time:.2f}秒)")
#     print(f"    链路分布数: {len(LINK_DISTRIBUTIONS_GLOBAL):,}")
    
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     # 步骤4：完成
#     # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     load_time = time_module.time() - start_time
    
#     print(f"\n[4/4] ✓ 数据加载和预处理完成！")
#     print(f"  总耗时: {load_time:.2f}秒")
#     print(f"    基础数据加载: {load_time - adj_time - dist_time:.2f}秒")
#     print(f"    邻接表构建: {adj_time:.2f}秒")
#     print(f"    链路分布计算: {dist_time:.2f}秒")
#     print(f"{'='*70}\n")
    
#     DATA_LOADED = True
    
#     return (G_GLOBAL, SPARSE_DATA_GLOBAL, NODE_TO_INDEX_GLOBAL,
#             SCENARIO_DATES_GLOBAL, SCENARIO_PROBS_GLOBAL, TIME_INTERVALS_PER_DAY_GLOBAL)


def _build_adjacency_lists(sparse_data:  Dict, node_to_index: Dict, n_scenarios: int) -> Tuple[Dict, Dict]:
    """
    构建正向和反向邻接表
    
    Args:
        sparse_data: 稀疏旅行时间数据
        node_to_index: 节点到索引的映射
        n_scenarios: 场景数量
    
    Returns:
        adj_list_forward: 正向邻接表 {from_node: [to_node1, to_node2, ...]}
        adj_list_backward: 反向邻接表 {to_node: [from_node1, from_node2, ...]}
    """
    index_to_node = {v:  k for k, v in node_to_index.items()}
    
    # 提取唯一边（去重）
    edges_set = set()
    for (scenario_idx, time_idx, from_idx, to_idx) in sparse_data.keys():
        if scenario_idx < n_scenarios:
            from_node = index_to_node[from_idx]
            to_node = index_to_node[to_idx]
            edges_set.add((from_node, to_node))
    
    # 构建邻接表
    adj_list_forward = defaultdict(list)
    adj_list_backward = defaultdict(list)
    
    for from_node, to_node in edges_set:
        adj_list_forward[from_node].append(to_node)
        adj_list_backward[to_node].append(from_node)
    
    # 转换为普通字典
    return dict(adj_list_forward), dict(adj_list_backward)

# def _precompute_link_distributions(sparse_data: Dict, node_to_index: Dict, n_scenarios: int) -> tuple:
#     """
#     预计算所有链路的旅行时间分布（同时生成正向和反向版本）
#     并生成边到时间片的索引，解决反向求解的冷启动性能问题。
#     """
#     print(f"    同时计算正向和反向链路分布...")
#     start_time = time_module.time()
    
#     # ... (导入逻辑保持不变) ...
#     try:
#         from forward_solver import LinkTimeDistribution as ForwardLinkDist
#         forward_available = True
#     except ImportError: 
#         print(f"      ⚠️ 警告: 无法导入 forward_solver.LinkTimeDistribution")
#         forward_available = False
    
#     try:
#         from reverse_solver_pseudocode import LinkTimeDistribution as ReverseLinkDist
#         reverse_available = True
#     except ImportError:
#         print(f"      ⚠️ 警告: 无法导入 reverse_solver_pseudocode.LinkTimeDistribution")
#         reverse_available = False
    
#     if not forward_available and not reverse_available:
#         raise ImportError("无法导入任何 LinkTimeDistribution 类")
    
#     index_to_node = {v: k for k, v in node_to_index.items()}
    
#     # 收集每条链路在每个时间片的旅行时间
#     link_time_data = defaultdict(list)
    
#     for (scenario_idx, time_idx, from_idx, to_idx), travel_time_minutes in sparse_data.items():
#         if scenario_idx >= n_scenarios: 
#             continue
        
#         from_node = index_to_node[from_idx]
#         to_node = index_to_node[to_idx]
#         travel_time_01min = int(travel_time_minutes * 10)  # 转换为0.1分钟单位
        
#         link_time_data[(from_node, to_node, time_idx)].append(travel_time_01min)
    
#     # 计算分布
#     link_distributions_forward = {}
#     link_distributions_backward = {}
    
#     # ✨ 新增：预计算边到可用时间片的映射
#     # key: (u, v), value: list of slots [0, 10, 20...]
#     edge_available_slots = defaultdict(list)
    
#     distribution_count = 0
#     skipped_count = 0
    
#     for (u, v, t), times in link_time_data.items():
#         # 统计频率
#         time_counts = defaultdict(int)
#         for time_val in times:
#             time_counts[time_val] += 1
        
#         # 计算概率
#         total = len(times)
#         time_prob = {time_val: count/total for time_val, count in time_counts.items()}
        
#         # 创建正向分布对象
#         if forward_available:
#             try: 
#                 link_distributions_forward[(u, v, t)] = ForwardLinkDist(time_prob, time_slot=t)
#                 distribution_count += 1
                
#                 # ✨ 记录该边在这个时间片有数据
#                 # 注意：只需要在 forward 或者 backward 其中一个记录即可，因为数据源是一样的
#                 edge_available_slots[(u, v)].append(t)
                
#             except (ValueError, Exception) as e:
#                 skipped_count += 1
        
#         # 创建反向分布对象
#         if reverse_available:
#             try: 
#                 link_distributions_backward[(u, v, t)] = ReverseLinkDist(time_prob, time_slot=t)
#                 # 如果 forward 不可用，这里也要记录
#                 if not forward_available:
#                     edge_available_slots[(u, v)].append(t)
#             except (ValueError, Exception) as e:
#                 pass

#     # ✨ 对时间片列表进行去重和排序
#     for key in edge_available_slots:
#         edge_available_slots[key] = sorted(list(set(edge_available_slots[key])))

#     elapsed = time_module.time() - start_time
    
#     print(f"      ✓ 完成 (用时 {elapsed:.2f}s)")
#     print(f"        正向分布:  {len(link_distributions_forward):,} 个")
#     print(f"        反向分布: {len(link_distributions_backward):,} 个")
#     print(f"        边索引: {len(edge_available_slots):,} 条边") # ✨
#     if skipped_count > 0:
#         print(f"        跳过无效:  {skipped_count} 个")
    
#     # ✨ 返回三个值
#     return link_distributions_forward, link_distributions_backward, edge_available_slots


def _precompute_link_distributions(sparse_data: Dict, node_to_index: Dict, n_scenarios: int) -> tuple:
    """
    预计算所有链路的旅行时间分布（同时生成正向和反向版本）
    并生成边到时间片的索引和旅行时间范围，解决反向求解的冷启动性能问题。
    """
    print(f"    同时计算正向和反向链路分布...")
    start_time = time_module.time()
    
    # ... (导入逻辑保持不变) ...
    try:
        from forward_solver import LinkTimeDistribution as ForwardLinkDist
        forward_available = True
    except ImportError: 
        print(f"      ⚠️ 警告: 无法导入 forward_solver.LinkTimeDistribution")
        forward_available = False
    
    try:
        from reverse_solver_pseudocode import LinkTimeDistribution as ReverseLinkDist
        reverse_available = True
    except ImportError:
        print(f"      ⚠️ 警告: 无法导入 reverse_solver_pseudocode.LinkTimeDistribution")
        reverse_available = False
    
    if not forward_available and not reverse_available:
        raise ImportError("无法导入任何 LinkTimeDistribution 类")
    
    index_to_node = {v: k for k, v in node_to_index.items()}
    
    # 收集每条链路在每个时间片的旅行时间
    link_time_data = defaultdict(list)
    
    for (scenario_idx, time_idx, from_idx, to_idx), travel_time_minutes in sparse_data.items():
        if scenario_idx >= n_scenarios: 
            continue
        
        from_node = index_to_node[from_idx]
        to_node = index_to_node[to_idx]
        travel_time_01min = int(travel_time_minutes * 10)  # 转换为0.1分钟单位
        
        link_time_data[(from_node, to_node, time_idx)].append(travel_time_01min)
    
    # 计算分布
    link_distributions_forward = {}
    link_distributions_backward = {}
    
    # ✨ 新增：预计算边到可用时间片的映射
    edge_available_slots = defaultdict(list)
    # ✨ 新增：每条边的全局(min, max) travel_time
    edge_travel_time_bounds = dict()  # (u,v) -> [min, max]
    
    distribution_count = 0
    skipped_count = 0
    
    for (u, v, t), times in link_time_data.items():
        # 统计频率
        time_counts = defaultdict(int)
        for time_val in times:
            time_counts[time_val] += 1
        
        # 计算概率
        total = len(times)
        time_prob = {time_val: count/total for time_val, count in time_counts.items()}
        
        # 创建正向分布对象
        if forward_available:
            try: 
                link_distributions_forward[(u, v, t)] = ForwardLinkDist(time_prob, time_slot=t)
                distribution_count += 1
                edge_available_slots[(u, v)].append(t)
            except (ValueError, Exception) as e:
                skipped_count += 1
        
        # 创建反向分布对象
        if reverse_available:
            try: 
                link_distributions_backward[(u, v, t)] = ReverseLinkDist(time_prob, time_slot=t)
                if not forward_available:
                    edge_available_slots[(u, v)].append(t)
            except (ValueError, Exception) as e:
                pass

        # ✨ 统计每条边的min/max travel time
        min_t = min(times)
        max_t = max(times)
        if (u, v) not in edge_travel_time_bounds:
            edge_travel_time_bounds[(u, v)] = [min_t, max_t]
        else:
            cur_min, cur_max = edge_travel_time_bounds[(u, v)]
            edge_travel_time_bounds[(u, v)][0] = min(cur_min, min_t)
            edge_travel_time_bounds[(u, v)][1] = max(cur_max, max_t)

    # ✨ 对时间片列表进行去重和排序
    for key in edge_available_slots:
        edge_available_slots[key] = sorted(list(set(edge_available_slots[key])))

    # ✨ 把min/max变为tuple更安全
    for k in edge_travel_time_bounds:
        edge_travel_time_bounds[k] = tuple(edge_travel_time_bounds[k])

    elapsed = time_module.time() - start_time
    
    print(f"      ✓ 完成 (用时 {elapsed:.2f}s)")
    print(f"        正向分布:  {len(link_distributions_forward):,} 个")
    print(f"        反向分布: {len(link_distributions_backward):,} 个")
    print(f"        边索引: {len(edge_available_slots):,} 条边")
    print(f"        边范围: {len(edge_travel_time_bounds):,} 条边 (min,max)")
    if skipped_count > 0:
        print(f"        跳过无效:  {skipped_count} 个")
    
    # ✨ 返回4个值
    return link_distributions_forward, link_distributions_backward,  edge_travel_time_bounds

def get_data():
    """获取全局基础数据"""
    if not DATA_LOADED:
        return load_data_once()
    return (G_GLOBAL, SPARSE_DATA_GLOBAL, NODE_TO_INDEX_GLOBAL,
            SCENARIO_DATES_GLOBAL, SCENARIO_PROBS_GLOBAL, TIME_INTERVALS_PER_DAY_GLOBAL)


# def get_precomputed_data():
#     """
#     ✨ 新增：获取预计算的数据结构
    
#     Returns:
#         adj_list_forward: 正向邻接表
#         adj_list_backward: 反向邻接表
#         link_distributions: 链路分布
#     """
#     if not DATA_LOADED:
#         raise RuntimeError("数据未加载，请先调用 load_data_once()")
    
#     return ADJ_LIST_FORWARD_GLOBAL, ADJ_LIST_BACKWARD_GLOBAL, LINK_DISTRIBUTIONS_GLOBAL


# ════════════════════════════════════════════════════════════════
# 保留原有的辅助函数
# ════════════════════════════════════════════════════════════════

def select_od_pair(node_to_index):
    """选择OD对（使用指定种子）"""
    nodes = list(node_to_index.keys())
    np.random.seed(int(time_module.time()))
    origin = np.random.choice(nodes)
    destination = np.random.choice([n for n in nodes if n != origin])
    return origin, destination


def time_to_string(time_01min):
    """将0.1分钟单位转换为HH:MM格式"""
    total_minutes = time_01min / 10
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    return f"{hours:02d}:{minutes:02d}"


def format_minutes(time_01min):
    """格式化分钟数（带单位）"""
    minutes = time_01min / 10
    return f"{minutes:.1f}分钟"


def format_path(path):
    """格式化路径输出"""
    if len(path) <= 10:
        return ' → '.join(map(str, path))
    else:
        return (f"{' → '.join(map(str, path[:5]))} → ..."
                f"→ {' → '.join(map(str, path[-3:]))}")


# ═══════════════════════════════════════════════════════════════════
# 测试1: 基本求解
# ═══════════════════════════════════════════════════════════════════

def test_1_basic_solve():
    """测试1: 基本求解"""
    print(f"\n{'='*70}")
    print(f"测试1: 基本求解")
    print(f"{'='*70}\n")
    
    # 获取全局数据
    adj_list_forward, adj_list_backward, link_distributions ,edge_travel_time_bounds= get_precomputed_data()
    G, sparse_data, node_to_index, scenario_dates, scenario_probs, time_intervals_per_day = get_data()
    
    # 初始化求解器
    mode = config.get_mode_config('standard')
    
    solver = ReverseLabelSettingSolver(
                G=G,
                sparse_data=sparse_data,
                node_to_index=node_to_index,
                scenario_dates=scenario_dates,
                scenario_probs=scenario_probs,
                time_intervals_per_day=time_intervals_per_day,
                L1=mode['L1'],
                L2=mode['L2'],
                adj_list=adj_list_forward,
                reverse_adj_list=adj_list_backward,
                link_distributions=link_dists_backward,
                edge_travel_time_bounds=edge_travel_time_bounds,
                K=K, verbose=config.REVERSE_VERBOSE
            )
    
    # 选择OD对（使用测试1的种子）
    origin, destination = select_od_pair(node_to_index)
    print(f"  测试OD对 (seed=1001): {origin} → {destination}")
    
    # 设置问题参数
    target_arrival_time = (config.DEFAULT_ARRIVAL_HOUR * 60 + 
                          config.DEFAULT_ARRIVAL_MINUTE) * 10
    alpha = config.REVERSE_ALPHA_DEFAULT
    
    print(f"  目标到达时间: {time_to_string(target_arrival_time)}")
    print(f"  可靠性要求: α={alpha}\n")
    
    # 求解
    result = solver.solve(
        origin=origin,
        destination=destination,
        target_arrival_time=target_arrival_time,
        alpha=alpha,
        max_labels=mode['max_labels']
    )
    
    # 验证结果
    print(f"\n{'─'*70}")
    print(f"测试1验证")
    print(f"{'─'*70}")
    
    assert result['success'], "❌ 求解失败"
    print(f"  ✓ 求解成功")
    
    assert result['path'] is not None, "❌ 路径为空"
    print(f"  ✓ 路径非空 (长度: {len(result['path'])})")
    
    assert result['path'][0] == origin, "❌ 起点不匹配"
    print(f"  ✓ 起点正确: {origin}")
    
    assert result['path'][-1] == destination, "❌ 终点不匹配"
    print(f"  ✓ 终点正确: {destination}")
    
    assert result['latest_departure_time'] > 0, "❌ 最晚出发时间无效"
    print(f"  ✓ 最晚出发时间: {time_to_string(result['latest_departure_time'])}")
    
    assert result['reserved_time'] > 0, "❌ 预留时间无效"
    print(f"  ✓ 预留时间: {result['reserved_time']/10:.1f}分钟")
    
    assert result['latest_departure_time'] < target_arrival_time, "❌ 出发晚于到达"
    print(f"  ✓ 时间逻辑正确")
    
    print(f"\n  🎉 测试1通过！")
    print(f"{'='*70}\n")
    
    return result


# ═══════════════════════════════════════════════════════════════════
# 测试2: α敏感性分析（完整版：0.05-0.95）
# ═══════════════════════════════════════════════════════════════════

def test_2_alpha_sensitivity():
    """测试2: α敏感性分析（0.05-0.95）"""
    print(f"\n{'='*70}")
    print(f"测试2: α敏感性分析（完整版）")
    print(f"{'='*70}\n")
    
    # 获取全局数据
    adj_list_forward, adj_list_backward, link_distributions ,edge_travel_time_bounds = get_precomputed_data()
    G, sparse_data, node_to_index, scenario_dates, scenario_probs, time_intervals_per_day = get_data()
    
    mode = config.get_mode_config('fast')
    
    solver = ReverseLabelSettingSolver(
                G=G,
                sparse_data=sparse_data,
                node_to_index=node_to_index,
                scenario_dates=scenario_dates,
                scenario_probs=scenario_probs,
                time_intervals_per_day=time_intervals_per_day,
                L1=mode['L1'],
                L2=mode['L2'],
                adj_list=adj_list_forward,
                reverse_adj_list=adj_list_backward,
                link_distributions=link_dists_backward,
                edge_travel_time_bounds=edge_travel_time_bounds,
                K=K, verbose=config.REVERSE_VERBOSE
            )
    
    # 使用测试2的种子
    origin, destination = select_od_pair(node_to_index)
    target_arrival_time = 9 * 60 * 10  # 09:00
    
    print(f"  测试OD对 (seed=2002): {origin} → {destination}")
    print(f"  目标到达时间: {time_to_string(target_arrival_time)}\n")
    
    # 完整的α值范围：0.05, 0.10, 0.15, ..., 0.95
    alphas = config.ALPHA_SENSITIVITY_VALUES
    
    print(f"  测试α值范围: 0.05 到 0.95 (步长0.05)")
    print(f"  总共 {len(alphas)} 个测试点\n")
    
    results = []
    # detailed_alphas = [0.50, 0.75, 0.95]  # 中、高、很高可靠性
    detailed_alphas = alphas
    detailed_results = {}
    
    print(f"  开始测试:")
    for i, alpha in enumerate(alphas, 1):
        print(f"    [{i:2d}/{len(alphas)}] α={alpha:.2f}...", end='', flush=True)
        
        save_all = alpha in detailed_alphas

        result = solver.solve(
            origin, destination, target_arrival_time, alpha,
            max_labels=mode['max_labels']
        )
        
        if result['success']:
            result_data = {
                'alpha': alpha,
                'latest_departure': result['latest_departure_time'],
                'expected_departure': result['expected_departure_time'],
                'reserved_time': result['reserved_time'],
                'path': result['path'],
                'path_length': len(result['path']),
                'target_arrival': target_arrival_time,
                'distribution': result['distribution']  # ← 保存分布用于可视化
            }
            
            # ✅ 如果保存了所有路径，添加到详细结果
            if save_all and 'all_paths' in result:
                result_data['all_paths'] = result['all_paths']
                result_data['num_candidates'] = result['num_candidate_paths']
                detailed_results[alpha] = result_data
                print(f" ✓ 最晚={time_to_string(result['latest_departure_time'])}, "
                      f"预留={result['reserved_time']/10:.1f}分, "
                      f"候选路径={result['num_candidate_paths']}")
            else:
                print(f" ✓ 最晚={time_to_string(result['latest_departure_time'])}, "
                      f"预留={result['reserved_time']/10:.1f}分")
            
            results.append(result_data)

                # 绘制所有候选路径的分布对比
            # plot_all_paths_distributions(
            #     result, 
            #     analysis_alpha, 
            #     target_arrival_time,
            #     output_file=f'result/alpha_{int(analysis_alpha*100)}_all_paths.png'
            # )
        else:
            print(f" ✗ 失败")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 验证
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n{'─'*70}")
    print(f"测试2验证")
    print(f"{'─'*70}")
    
    success_rate = len(results) / len(alphas) * 100
    print(f"  成功率: {len(results)}/{len(alphas)} ({success_rate:.1f}%)")
    
    assert len(results) >= len(alphas) * 0.7, \
        f"❌ 成功求解的α值太少: {len(results)}/{len(alphas)}"
    print(f"  ✓ 成功率达标 (≥70%)")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 打印详细对比表（全部使用HH:MM格式）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n  α敏感性详细对比:")
    print(f"  {'α值':<8} {'最晚出发':<12} {'期望出发':<12} {'目标到达':<12} "
          f"{'预留(分)':<12} {'路径长度':<10}")
    print(f"  {'-'*80}")
    
    # 显示所有结果（或部分关键点）
    display_all = len(results) <= 10
    
    if display_all:
        for r in results:
            print(f"  {r['alpha']:<8.2f} "
                  f"{time_to_string(r['latest_departure']):<12} "
                  f"{time_to_string(r['expected_departure']):<12} "
                  f"{time_to_string(r['target_arrival']):<12} "
                  f"{r['reserved_time']/10:<12.1f} "
                  f"{r['path_length']:<10}")
    else:
        # 显示关键点
        key_indices = [0, len(results)//4, len(results)//2, 3*len(results)//4, -1]
        for i in key_indices:
            if i < len(results):
                r = results[i]
                print(f"  {r['alpha']:<8.2f} "
                      f"{time_to_string(r['latest_departure']):<12} "
                      f"{time_to_string(r['expected_departure']):<12} "
                      f"{time_to_string(r['target_arrival']):<12} "
                      f"{r['reserved_time']/10:<12.1f} "
                      f"{r['path_length']:<10}")
        print(f"  ...(显示 {len(key_indices)}/{len(results)} 个结果，完整结果见输出文件)")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 打印路径详情（选择几个代表性的α值）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n  路径详情（代表性α值）:")
    print(f"  {'-'*70}")
    
    # 选择3个代表性α值：低、中、高
    representative_indices = []
    if len(results) > 0:
        representative_indices.append(0)
    if len(results) >= 2:
        representative_indices.append(len(results)//2)
    if len(results) >= 3:
        representative_indices.append(-1)
    
    for idx in representative_indices:
        if idx < len(results):
            r = results[idx]
            print(f"\n  【α = {r['alpha']:.2f}】")
            print(f"    起点: {origin}")
            print(f"    终点: {destination}")
            print(f"    路径: {format_path(r['path'])}")
            print(f"    路径长度: {r['path_length']} 个节点")
            print(f"    ┌─ 时间信息 ─────────────────────────────────────────┐")
            print(f"    │ 最晚出发时间: {time_to_string(r['latest_departure']):<10} "
                  f"({format_minutes(r['latest_departure'])})  │")
            print(f"    │ 期望出发时间: {time_to_string(r['expected_departure']):<10} "
                  f"({format_minutes(r['expected_departure'])})  │")
            print(f"    │ 目标到达时间: {time_to_string(r['target_arrival']):<10} "
                  f"({format_minutes(r['target_arrival'])})  │")
            print(f"    │ 预留时间:     {r['reserved_time']/10:>6.1f} 分钟"
                  f"{' '*26}│")
            print(f"    └────────────────────────────────────────────────────┘")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 验证单调性
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n  单调性检查 (抽样验证):")
    monotonic_violations = 0
    check_indices = [i for i in range(len(results)-1) if i % 3 == 0]
    
    for i in check_indices:
        curr = results[i]
        next_r = results[i+1]
        
        # α增大时，最晚出发时间应该减小或相近（容差10分钟）
        if curr['latest_departure'] < next_r['latest_departure'] - 100:
            monotonic_violations += 1
            print(f"    ⚠ α={curr['alpha']:.2f}→{next_r['alpha']:.2f}: "
                  f"最晚出发反而增加 "
                  f"({time_to_string(curr['latest_departure'])} → "
                  f"{time_to_string(next_r['latest_departure'])})")
    
    if monotonic_violations == 0:
        print(f"    ✓ 所有检查点符合单调性")
    else:
        print(f"    ⚠ {monotonic_violations}/{len(check_indices)} 个点违反单调性 "
              f"({monotonic_violations/len(check_indices)*100:.1f}%)")
    
    try:
        save_alpha_sensitivity_results(results, origin, destination, target_arrival_time)
        print(f"\n  ✓ 详细结果已保存到: alpha_sensitivity_results.txt")
    except Exception as e:
        print(f"\n  ⚠ 保存结果失败: {e}")
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 可视化
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if config.SHOW_PLOTS and len(results) >= 5:
        try:
            plot_alpha_sensitivity(results, target_arrival_time)
        except Exception as e:
            print(f"    ⚠ 可视化失败: {e}")
    
    print(f"\n  🎉 测试2通过！")
    print(f"{'='*70}\n")
    
    return {
        'all_results': results,
        'detailed_results': detailed_results,  # ← 新增：包含所有候选路径的详细结果
        'origin': origin,
        'destination': destination,
        'target_arrival_time': target_arrival_time
    }

def save_alpha_sensitivity_results(results, origin, destination, target_arrival_time):
    """保存α敏感性分析详细结果到文件（全部使用HH:MM格式）"""
    filename = 'result/alpha_sensitivity_results.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*90 + "\n")
        f.write("α敏感性分析详细结果\n")
        f.write("="*90 + "\n\n")
        
        f.write(f"起点: {origin}\n")
        f.write(f"终点: {destination}\n")
        f.write(f"目标到达时间: {time_to_string(target_arrival_time)} "
                f"({format_minutes(target_arrival_time)})\n")
        f.write(f"测试α值数量: {len(results)}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n" + "="*90 + "\n\n")
        
        # 汇总表格
        f.write("【汇总表格】\n")
        f.write("-"*90 + "\n")
        f.write(f"{'α值':<8} {'最晚出发':<12} {'期望出发':<12} {'目标到达':<12} "
                f"{'预留(分)':<12} {'路径长度':<10}\n")
        f.write("-"*90 + "\n")
        
        for r in results:
            f.write(f"{r['alpha']:<8.2f} "
                    f"{time_to_string(r['latest_departure']):<12} "
                    f"{time_to_string(r['expected_departure']):<12} "
                    f"{time_to_string(r['target_arrival']):<12} "
                    f"{r['reserved_time']/10:<12.1f} "
                    f"{r['path_length']:<10}\n")
        
        # 详细路径信息
        f.write("\n" + "="*90 + "\n")
        f.write("【详细路径信息】\n")
        f.write("="*90 + "\n")
        
        for r in results:
            f.write(f"\n{'─'*90}\n")
            f.write(f"α = {r['alpha']:.2f}\n")
            f.write(f"{'─'*90}\n")
            
            f.write(f"路径摘要: {format_path(r['path'])}\n")
            f.write(f"完整路径: {' → '.join(map(str, r['path']))}\n")
            f.write(f"路径长度: {r['path_length']} 个节点\n\n")
            
            f.write(f"时间信息:\n")
            f.write(f"  最晚出发时间: {time_to_string(r['latest_departure'])} "
                    f"({format_minutes(r['latest_departure'])})\n")
            f.write(f"  期望出发时间: {time_to_string(r['expected_departure'])} "
                    f"({format_minutes(r['expected_departure'])})\n")
            f.write(f"  目标到达时间: {time_to_string(r['target_arrival'])} "
                    f"({format_minutes(r['target_arrival'])})\n")
            f.write(f"  预留时间:     {r['reserved_time']/10:.1f} 分钟\n")  # 修正这里
            
            # 时间差异分析
            time_diff = r['expected_departure'] - r['latest_departure']
            f.write(f"  出发时间差异: {time_diff/10:.1f} 分钟 "
                    f"(期望 - 最晚)\n")
        
        # 统计信息
        f.write("\n" + "="*90 + "\n")
        f.write("【统计信息】\n")
        f.write("="*90 + "\n\n")
        
        reserved_times = [r['reserved_time']/10 for r in results]
        path_lengths = [r['path_length'] for r in results]
        
        f.write(f"预留时间统计:\n")
        f.write(f"  最小值: {min(reserved_times):.1f} 分钟 (α={results[np.argmin(reserved_times)]['alpha']:.2f})\n")
        f.write(f"  最大值: {max(reserved_times):.1f} 分钟 (α={results[np.argmax(reserved_times)]['alpha']:.2f})\n")
        f.write(f"  平均值: {np.mean(reserved_times):.1f} 分钟\n")
        f.write(f"  标准差: {np.std(reserved_times):.1f} 分钟\n\n")  # 修正这里
        
        f.write(f"路径长度统计:\n")
        f.write(f"  最小值: {min(path_lengths)} 个节点\n")
        f.write(f"  最大值: {max(path_lengths)} 个节点\n")
        f.write(f"  平均值: {np.mean(path_lengths):.1f} 个节点\n")
        
        # 单调性分析
        f.write(f"\n单调性分析:\n")
        violations = 0
        for i in range(len(results)-1):
            if results[i]['latest_departure'] < results[i+1]['latest_departure'] - 100:
                violations += 1
        
        f.write(f"  检查点数: {len(results)-1}\n")
        f.write(f"  违反单调性: {violations} 个\n")
        f.write(f"  单调性率: {(1-violations/(len(results)-1))*100:.1f}%\n")

def time_to_string(time_01min):
    """
    将0.1分钟单位转换为HH:MM格式
    
    Args:
        time_01min: 时间（0.1分钟单位）
        
    Returns:
        HH:MM格式的字符串
    """
    total_minutes = time_01min / 10
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    return f"{hours:02d}:{minutes:02d}"


def format_minutes(time_01min):
    """
    格式化分钟数（带单位）
    
    Args:
        time_01min: 时间（0.1分钟单位）
        
    Returns:
        格式化的字符串，如 "505.0分钟"
    """
    minutes = time_01min / 10
    return f"{minutes:.1f}分钟"


def format_path(path):
    """格式化路径输出"""
    if len(path) <= 10:
        return ' → '.join(map(str, path))
    else:
        return (f"{' → '.join(map(str, path[:5]))} → ..."
                f"→ {' → '.join(map(str, path[-3:]))}")

# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════

def time_to_string(time_01min):
    """
    将0.1分钟单位转换为HH:MM格式
    
    Args:
        time_01min: 时间（0.1分钟单位）
        
    Returns:
        HH:MM格式的字符串
    """
    total_minutes = time_01min / 10
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    return f"{hours:02d}:{minutes:02d}"


def plot_alpha_sensitivity(results, target_arrival_time):
    """绘制α敏感性分析图"""
    alphas = [r['alpha'] for r in results]
    latest_deps = [r['latest_departure']/10 for r in results]
    reserved_times = [r['reserved_time']/10 for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 子图1: 最晚出发时间
    ax1.plot(alphas, latest_deps, 'b-o', linewidth=2, markersize=4, label='Latest Departure')
    ax1.axhline(target_arrival_time/10, color='orange', linestyle='--', 
                linewidth=2, label='Target Arrival')
    ax1.set_xlabel('Reliability α', fontsize=12)
    ax1.set_ylabel('Departure Time (minutes)', fontsize=12)
    ax1.set_title('α Sensitivity - Departure Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 子图2: 预留时间
    ax2.plot(alphas, reserved_times, 'r-s', linewidth=2, markersize=4, label='Reserved Time')
    ax2.set_xlabel('Reliability α', fontsize=12)
    ax2.set_ylabel('Reserved Time (minutes)', fontsize=12)
    ax2.set_title('α Sensitivity - Reserved Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('result/alpha_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ 可视化已保存: alpha_sensitivity_analysis.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 测试3: 性能测试
# ═══════════════════════════════════════════════════════════════════

def test_3_performance():
    """测试3: 性能测试"""
    print(f"\n{'='*70}")
    print(f"测试3: 性能测试")
    print(f"{'='*70}\n")
    
    # 获取全局数据
    adj_list_forward, adj_list_backward, link_distributions ,edge_travel_time_bounds = get_precomputed_data()
    G, sparse_data, node_to_index, scenario_dates, scenario_probs, time_intervals_per_day = get_data()
    
    # 使用测试3的种子
    origin, destination = select_od_pair(node_to_index)
    target_arrival_time = 9 * 60 * 10
    
    print(f"  测试OD对 (seed=3003): {origin} → {destination}")
    print(f"  目标到达: {time_to_string(target_arrival_time)}\n")
    
    # 测试不同配置
    test_configs = [
        ('快速模式', config.FAST_MODE),
        ('标准模式', config.STANDARD_MODE),
    ]
    
    performance_results = []
    
    for config_name, mode in test_configs:
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  配置: {config_name}")
        print(f"    L1={mode['L1']}, L2={mode['L2']}, 最大标签={mode['max_labels']:,}")
        
        solver = ReverseLabelSettingSolver(
                G=G,
                sparse_data=sparse_data,
                node_to_index=node_to_index,
                scenario_dates=scenario_dates,
                scenario_probs=scenario_probs,
                time_intervals_per_day=time_intervals_per_day,
                L1=mode['L1'],
                L2=mode['L2'],
                adj_list=adj_list_forward,
                reverse_adj_list=adj_list_backward,
                link_distributions=link_dists_backward,
                edge_travel_time_bounds=edge_travel_time_bounds,
                 K=K, verbose=config.REVERSE_VERBOSE
            )
        
        start = time_module.time()
        result = solver.solve(
            origin, destination, target_arrival_time, 0.95,
            max_labels=mode['max_labels']
        )
        elapsed = time_module.time() - start
        
        if result['success']:
            perf_data = {
                'config': config_name,
                'L1': mode['L1'],
                'L2': mode['L2'],
                'time': elapsed,
                'iterations': result['iterations'],
                'labels_generated': result['stats']['labels_generated'],
                'labels_dominated': result['stats']['labels_dominated'],
                'pruning_rate': result['stats']['labels_dominated']/result['stats']['labels_generated']*100
            }
            performance_results.append(perf_data)
            
            print(f"    ✓ 成功")
            print(f"      耗时: {elapsed:.2f}秒")
            print(f"      迭代: {result['iterations']}")
            print(f"      生成标签: {result['stats']['labels_generated']:,}")
            print(f"      剪枝率: {perf_data['pruning_rate']:.1f}%")
            print(f"      最晚出发: {time_to_string(result['latest_departure_time'])}")
        else:
            print(f"    ✗ 失败")
    
    # 性能对比
    if len(performance_results) >= 2:
        print(f"\n  性能对比:")
        fast = performance_results[0]
        standard = performance_results[1]
        speedup = standard['time'] / fast['time']
        print(f"    快速模式 vs 标准模式:")
        print(f"      速度提升: {speedup:.2f}x")
        print(f"      标签数对比: {fast['labels_generated']:,} vs {standard['labels_generated']:,}")
    
    print(f"\n  🎉 测试3完成！")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════
# 测试4: 时间一致性
# ═══════════════════════════════════════════════════════════════════

def test_4_time_consistency():
    """测试4: 时间一致性"""
    print(f"\n{'='*70}")
    print(f"测试4: 时间一致性检查")
    print(f"{'='*70}\n")
    
    # 获取全局数据
    G, sparse_data, node_to_index, scenario_dates, scenario_probs, time_intervals_per_day = get_data()
    adj_list_forward, adj_list_backward, link_distributions,edge_travel_time_bounds = get_precomputed_data()
    
    mode = config.get_mode_config('fast')
    
    solver = ReverseLabelSettingSolver(
                G=G,
                sparse_data=sparse_data,
                node_to_index=node_to_index,
                scenario_dates=scenario_dates,
                scenario_probs=scenario_probs,
                time_intervals_per_day=time_intervals_per_day,
                L1=mode['L1'],
                L2=mode['L2'],
                adj_list=adj_list_forward,
                reverse_adj_list=adj_list_backward,
                link_distributions=link_dists_backward,
                edge_travel_time_bounds=edge_travel_time_bounds,
                K=K, verbose=config.REVERSE_VERBOSE
            )
    
    # 使用测试4的种子
    origin, destination = select_od_pair(node_to_index)
    
    print(f"  测试OD对 (seed=4004): {origin} → {destination}\n")
    
    # 测试不同到达时间
    test_times = config.TIME_BUDGET_TEST_TIMES[:3]
    results = []
    
    for hour, minute in test_times:
        target_time = (hour * 60 + minute) * 10
        time_str = f"{hour:02d}:{minute:02d}"
        
        print(f"  测试目标到达: {time_str}...", end='', flush=True)
        
        result = solver.solve(
            origin, destination, target_time, 0.95,
            max_labels=mode['max_labels']
        )
        
        if result['success']:
            results.append({
                'target_time': target_time,
                'time_str': time_str,
                'latest_dep': result['latest_departure_time'],
                'reserved': result['reserved_time']
            })
            print(f" ✓ 最晚出发={time_to_string(result['latest_departure_time'])}")
        else:
            print(f" ✗ 失败")
    
    # 验证
    print(f"\n{'─'*70}")
    print(f"测试4验证")
    print(f"{'─'*70}")
    
    for r in results:
        # 验证时间逻辑
        assert r['latest_dep'] < r['target_time'], \
            f"❌ 时间逻辑错误: {r['time_str']}"
        print(f"  ✓ {r['time_str']}: 时间逻辑正确")
        
        # 验证预留时间
        expected_reserved = r['target_time'] - r['latest_dep']
        assert abs(r['reserved'] - expected_reserved) < 1, \
            f"❌ 预留时间计算错误: {r['time_str']}"
        print(f"    预留时间: {r['reserved']/10:.1f}分")
    
    print(f"\n  🎉 测试4通过！")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════
# 测试5: 多OD对测试
# ═══════════════════════════════════════════════════════════════════

def test_5_multiple_od_pairs():
    """测试5: 多OD对测试（测试算法稳定性）- 修改版"""
    print(f"\n{'='*70}")
    print(f"测试5: 多OD对测试")
    print(f"{'='*70}\n")
    
    # 获取全局数据
    G, sparse_data, node_to_index, scenario_dates, scenario_probs, time_intervals_per_day = get_data()
    adj_list_forward, adj_list_backward, link_distributions ,edge_travel_time_bounds= get_precomputed_data()
    
    mode = config.get_mode_config('fast')
    
    solver = ReverseLabelSettingSolver(
                G=G,
                sparse_data=sparse_data,
                node_to_index=node_to_index,
                scenario_dates=scenario_dates,
                scenario_probs=scenario_probs,
                time_intervals_per_day=time_intervals_per_day,
                L1=mode['L1'],
                L2=mode['L2'],
                adj_list=adj_list_forward,
                reverse_adj_list=adj_list_backward,
                link_distributions=link_dists_backward,
                edge_travel_time_bounds=edge_travel_time_bounds,
                K=K, verbose=config.REVERSE_VERBOSE
            )
    
    target_arrival_time = 9 * 60 * 10
    alpha = 0.95
    
    print(f"  测试多个不同的OD对")
    print(f"  目标到达: {time_to_string(target_arrival_time)}, α={alpha}\n")
    
    # 测试5对不同的OD
    num_tests = config.NUM_TESTS
    success_count = 0
    results = []
    
    for i in range(num_tests):
        seed = 5000 + i
        origin, destination = select_od_pair(node_to_index)
        
        print(f"  测试 {i+1}/{num_tests} (seed={seed}): {origin}→{destination}...", 
              end='', flush=True)
        
        result = solver.solve(
            origin, destination, target_arrival_time, alpha,
            max_labels=mode['max_labels']
        )
        
        if result['success']:
            success_count += 1
            # ✅ 修改：返回完整数据
            results.append({
                'od': (origin, destination),
                'origin': origin,  # ← 新增
                'destination': destination,  # ← 新增
                'latest_dep': result['latest_departure_time'],
                'expected_dep': result['expected_departure_time'],  # ← 新增
                'reserved': result['reserved_time'],
                'path': result['path'],  # ← 新增
                'path_length': len(result['path']),
                'target_arrival': target_arrival_time,  # ← 新增
                'alpha': alpha,  # ← 新增
                'distribution': result['distribution']  # ← 新增（用于可视化）
            })
            print(f" ✓ 预留={result['reserved_time']/10:.1f}分, 路径={len(result['path'])}节点")
        else:
            print(f" ✗ 失败")
    
    # 验证
    print(f"\n{'─'*70}")
    print(f"测试5验证")
    print(f"{'─'*70}")
    
    success_rate = success_count / num_tests * 100
    print(f"  成功率: {success_count}/{num_tests} ({success_rate:.1f}%)")
    
    assert success_count >= num_tests * 0.6, \
        f"❌ 成功率太低: {success_rate:.1f}%"
    print(f"  ✓ 成功率达标 (≥60%)")
    
    if results:
        print(f"\n  结果统计:")
        reserved_times = [r['reserved']/10 for r in results]
        path_lengths = [r['path_length'] for r in results]
        print(f"    预留时间: 均值={np.mean(reserved_times):.1f}分, "
              f"标准差={np.std(reserved_times):.1f}分")
        print(f"    路径长度: 均值={np.mean(path_lengths):.1f}, "
              f"范围=[{min(path_lengths)}, {max(path_lengths)}]")
    
    print(f"\n  🎉 测试5通过！")
    print(f"{'='*70}\n")
    
    return results  # ← 返回完整结果


# ═══════════════════════════════════════════════════════════════════
# 运行所有测试
# ═══════════════════════════════════════════════════════════════════

def run_all_tests():
    """运行所有测试"""
    print(f"\n{'='*70}")
    print(f"反向求解器测试套件（优化版）")
    print(f"{'='*70}")
    print(f"  日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  配置: L1={config.REVERSE_L1}, L2={config.REVERSE_L2}")
    print(f"  优化: 全局加载数据 + 独立随机种子")
    print(f"{'='*70}\n")
    
    # 预先加载数据
    load_data_once()
    
    start_time = time_module.time()
    
    try:
        # 测试1: 基本求解
        test_1_basic_solve()
        
        # 测试2: α敏感性（完整版）
        test_2_alpha_sensitivity()
        
        # 测试3: 性能
        test_3_performance()
        
        # 测试4: 时间一致性
        test_4_time_consistency()
        
        # 测试5: 多OD对
        test_5_multiple_od_pairs()
        
        total_time = time_module.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"所有测试完成！✓")
        print(f"{'='*70}")
        print(f"  总耗时: {total_time:.2f}秒")
        print(f"  状态: 全部通过 ✓")
        print(f"  数据加载: 仅一次（优化）")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"测试失败！✗")
        print(f"{'='*70}")
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}\n")
        return False

# 在文件末尾添加
from visualization_generator import generate_html_with_svg

def run_all_tests_with_visualization(testname: str ):
    """运行所有测试并生成可视化（修改版）"""
    print(f"\n{'='*70}")
    print(f"反向求解器测试套件（带可视化）")
    print(f"{'='*70}\n")
    
    # 预先加载数据
    load_data_once()
    
    start_time = time_module.time()
    
    # 存储所有结果
    results_all = {}
    
    try:
        # 运行测试1
        print("运行测试1...")
        results_all['test1'] = test_1_basic_solve()
        
        # 运行测试2（增强版）
        print("运行测试2...")
        results_all['test2'] = test_2_alpha_sensitivity()
        
        # 运行测试3
        print("运行测试3...")
        results_all['test3'] = []  # 可选
        
        # 运行测试5（修改版）
        print("运行测试5...")
        results_all['test5'] = test_5_multiple_od_pairs()
        
        total_time = time_module.time() - start_time
        
        print(f"\n所有测试完成！总耗时: {total_time:.2f}秒")
        # ✅ 保存结果
        print("\n保存测试结果...")
        save_results(results_all, solver_type='reverse', output_dir=f'results/{testname}')

        
        return True
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
# ═══════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    
    # 验证配置
    print("验证配置...")
    config.validate_config()
    print()
    
    if len(sys.argv) > 1:
        # 预先加载数据
        load_data_once()
        
        # 运行指定测试
        test_name = sys.argv[1]
        if test_name == '1':
            test_1_basic_solve()
        elif test_name == '2':
            test_2_alpha_sensitivity()
        elif test_name == '3':
            test_3_performance()
        elif test_name == '4':
            test_4_time_consistency()
        elif test_name == '5':
            test_5_multiple_od_pairs()
        else:
            print(f"未知测试: {test_name}")
            print(f"可用测试: 1, 2, 3, 4, 5")
    else:
        # 运行所有测试并生成可视化
        success = run_all_tests_with_visualization()
        sys.exit(0 if success else 1)