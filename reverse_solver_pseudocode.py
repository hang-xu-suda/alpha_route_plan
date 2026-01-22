"""
反向Label-Setting求解器 - 严格按照伪代码实现（完整版）
包含详细的概率计算说明
"""

import numpy as np
import heapq
import time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════
# 数据结构定义
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AlphaDiscreteDistribution:
    """α-离散分布类"""
    values: np.ndarray
    L1:  int
    weights: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """后初始化"""
        if not isinstance(self.values, np.ndarray):
            self.values = np.array(self.values)
        
        if self.weights is None:
            self.weights = np.ones(len(self.values)) / len(self.values)
        elif not isinstance(self.weights, np.ndarray):
            self.weights = np.array(self.weights)
        
        if len(self.weights) > 0 and self.weights.sum() > 0:
            self.weights = self.weights / self.weights.sum()
    
    def get_quantile(self, alpha: float) -> float:
        """计算α分位数"""
        if len(self.values) == 0:
            return float('inf')
        if alpha <= 0:
            return float(self.values.min())
        if alpha >= 1:
            return float(self.values.max())
        
        sorted_idx = np.argsort(self.values)
        sorted_vals = self.values[sorted_idx]
        sorted_wts = self.weights[sorted_idx]
        cumsum = np.cumsum(sorted_wts)
        
        return float(np.interp(alpha, cumsum, sorted_vals))
    
    def get_mean(self) -> float:
        """计算均值"""
        return float(np.sum(self.values * self.weights))
    
    def get_expected(self) -> float:
        """计算期望值（与 get_mean 相同）"""
        return self.get_mean()
    
    def get_variance(self) -> float:
        """计算方差"""
        mu = self.get_mean()
        return float(np.sum(self.weights * (self.values - mu) ** 2))
    
    def get_std(self) -> float:
        """计算标准差"""
        return float(np.sqrt(self.get_variance()))
    
    def get_median(self) -> float:
        return np.median(self.values)
    
    
    def reverse_convolve(self,
                    get_link_dist_func,
                    link_time_ranges,
                    predecessor:  int,
                    current:  int,
                    time_intervals_per_day: int,
                    L2:  int,
                    verbose: bool = False) -> 'AlphaDiscreteDistribution':
        """
        反向卷积（基于精确概率计算）
        
        Args:
            get_link_dist_func: 获取链路分布的函数
            predecessor: 前驱节点u
            current: 当前节点v
            time_intervals_per_day: 每天时间片数
            L2: 未使用（保持接口兼容）
            verbose: 是否输出详细信息
            
        Returns:
            出发时间分布 A(u)
        """
        

        uv_range = link_time_ranges.get((predecessor, current))
        if not uv_range:
            raise ValueError(f"缺少边 {predecessor}->{current} 的范围数据")
            
        min_travel, max_travel = uv_range
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤2: 确定出发时间搜索范围
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        min_arrival = float(np.min(self.values))
        max_arrival = float(np.max(self.values))
        
        min_departure = min_arrival - max_travel
        max_departure = max_arrival - min_travel
        
        # 离散化步长（1分钟 = 10个单位）
        step = 1
        
        # 生成候选出发时间
        candidate_departures = np.arange(
            int(np.floor(min_departure)),
            int(np.ceil(max_departure)) + step,
            step
        )
        
        if verbose:
            print(f"    出发时间范围: [{min_departure:.1f}, {max_departure:.1f}]")
            print(f"    候选出发时间数: {len(candidate_departures)}")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤3: 构建到达时间的概率分布（✅ 使用实际权重）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        arrival_probs = {}
        
        for i, t_arr in enumerate(self.values):
            # ✅ 使用实际权重而不是假设等权重
            weight = self.weights[i] if self.weights is not None else 1.0 / self.L1
            
            t_arr_int = int(round(t_arr))
            if t_arr_int not in arrival_probs:
                arrival_probs[t_arr_int] = 0.0
            arrival_probs[t_arr_int] += weight
        
        # 归一化到达时间概率
        total_arrival_prob = sum(arrival_probs.values())
        if total_arrival_prob > 0:
            for t in arrival_probs:
                arrival_probs[t] /= total_arrival_prob
        
        if verbose:
            print(f"    到达时间分布大小: {len(arrival_probs)}")
            print(f"    到达时间概率和: {sum(arrival_probs.values()):.6f}")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤4: 对每个候选出发时间，计算其概率
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        departure_probs={}
        candidate_departures = np.array(candidate_departures)
        arr_times = np.array(list(arrival_probs.keys()))
        arr_probs = np.array(list(arrival_probs.values()))
        dep_probs_raw = np.zeros(candidate_departures.shape, dtype=float)
        slot_time=0.1
        for i, t_dep in enumerate(candidate_departures):
            slot_dep = int(t_dep ) % time_intervals_per_day
            D_slot = get_link_dist_func(predecessor, current, slot_dep)
            required_travels = arr_times - t_dep    
            prob_travels = D_slot.get_probabilities_batch(required_travels)
            dep_probs_raw[i] = np.dot(arr_probs, prob_travels)
        mask = dep_probs_raw > 1e-10
        departure_probs = dict(zip(candidate_departures[mask], dep_probs_raw[mask]))
        
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤5: 归一化并构造新分布
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        if not departure_probs:
            if verbose:
                print(f"    警告:  未生成任何有效的出发概率")
                print(f"    候选数: {len(candidate_departures)}, 到达数: {len(arrival_probs)}")
            
            # 返回一个基于最早出发时间的默认分布
            default_time = min_departure
            return AlphaDiscreteDistribution(
                values=np.array([default_time]),
                L1=1,
                weights=np.array([1.0])
            )
        
        # 计算总概率
        total_prob = sum(departure_probs.values())
        
        if total_prob <= 1e-10:
            if verbose:
                print(f"    警告: 总概率过小:  {total_prob:.10e}")
            
            # 使用未归一化的分布
            times = np.array(sorted(departure_probs.keys()))
            weights = np.ones(len(times)) / len(times)
            return AlphaDiscreteDistribution(
                values=times[: self.L1] if len(times) > self.L1 else times,
                L1=min(len(times), self.L1),
                weights=weights[: self.L1] if len(weights) > self.L1 else weights
            )
        
        # ✅ 一次性归一化（避免重复）
        for t in departure_probs:
            departure_probs[t] /= total_prob
        
        # 转换为数组
        times = np.array(sorted(departure_probs.keys()), dtype=float)
        probs = np.array([departure_probs[t] for t in times], dtype=float)
        
        # 验证概率和
        prob_sum = probs.sum()
        if abs(prob_sum - 1.0) > 1e-6:
            if verbose:
                print(f"    警告:  概率和 = {prob_sum:.6f} != 1.0，重新归一化")
            probs = probs / prob_sum
        
        if verbose:
            print(f"    出发时间分布大小:   {len(times)}")
            print(f"    概率和:  {probs.sum():.6f}")
        
        # ═══════════════════════════════════════════════════════════════
        # 确定性分位数采样（保留分布形状）
        # ═══════════════════════════════════════════════════════════════
        
        if len(times) > self.L1:
            # 需要降采样
            cdf = np.cumsum(probs)
            target_quantiles = np.linspace(1/(self.L1 + 1), self.L1/(self.L1 + 1), self.L1)
            sampled_times = np.interp(target_quantiles, cdf, times)
            sampled_weights = np.ones(self.L1) / self.L1
            
            if verbose:
                print(f"    降采样:  {len(times)} -> {self.L1} 个点")
        
        elif len(times) < self.L1:
            # 需要上采样
            cdf = np.cumsum(probs)
            target_quantiles = np.linspace(1/(self.L1 + 1), self.L1/(self.L1 + 1), self.L1)
            sampled_times = np.interp(target_quantiles, cdf, times)
            sampled_weights = np.ones(self.L1) / self.L1
            
            if verbose:
                print(f"    上采样: {len(times)} -> {self.L1} 个点")
        
        else:
            # 大小刚好
            sampled_times = times
            sampled_weights = probs  # ✅ 使用实际概率而不是均匀权重
            
            if verbose:
                print(f"    保持原始大小: {self.L1} 个点")
        
        # 确保有序
        sorted_indices = np.argsort(sampled_times)
        sampled_times = sampled_times[sorted_indices]
        sampled_weights = sampled_weights[sorted_indices]
        
        # 最终归一化（确保权重和为1）
        sampled_weights = sampled_weights / sampled_weights.sum()
        
        if verbose: 
            print(f"    最终分布:")
            print(f"      大小: {len(sampled_times)}")
            print(f"      时间范围: [{sampled_times[0]:.1f}, {sampled_times[-1]:.1f}]")
            print(f"      权重和: {sampled_weights.sum():.6f}")
            print(f"      期望值: {np.sum(sampled_times * sampled_weights):.2f}")
        
        return AlphaDiscreteDistribution(
            values=sampled_times,
            L1=self.L1,
            weights=sampled_weights
        )



    def _find_nearest_slot(self, target_slot: int, available_slots: List[int],
                        time_intervals_per_day: int) -> int:
        """找到最近的时间片"""
        min_dist = float('inf')
        best_slot = available_slots[0]
        
        for slot in available_slots:
            dist = abs(slot - target_slot)
            cyclic_dist = min(dist, time_intervals_per_day - dist)
            
            if cyclic_dist < min_dist:
                min_dist = cyclic_dist
                best_slot = slot
        
        return best_slot
    
    # def _get_available_slots(self, get_link_dist_func, u: int, v: int) -> List[int]:
    #     if not hasattr(AlphaDiscreteDistribution, '_slot_cache'):
    #         AlphaDiscreteDistribution._slot_cache = {}
    #     cache_key = (u, v)
    #     if cache_key in AlphaDiscreteDistribution._slot_cache:
    #         return AlphaDiscreteDistribution._slot_cache[cache_key]
    #     available = []
    #     try:
    #         link_distributions = get_link_dist_func.__self__.link_distributions
    #         for (link_u, link_v, slot) in link_distributions.keys():
    #             if link_u == u and link_v == v:
    #                 available.append(slot)
    #     except AttributeError:
    #         raise ValueError("无法访问链路分布数据")
    #     result = sorted(set(available))
    #     AlphaDiscreteDistribution._slot_cache[cache_key] = result
    #     return result

    # def _get_available_slots(self, get_link_dist_func, u: int, v: int) -> List[int]:
    #     """
    #     获取该边可用的时间片列表
    #     优化：优先使用预计算的索引，避免全表扫描导致的冷启动慢问题。
    #     """
    #     # 1. 尝试从求解器实例获取预计算索引 (最快路径)
    #     try:
    #         solver = getattr(get_link_dist_func, '__self__', None)
    #         if solver and hasattr(solver, 'edge_available_slots'):
    #             # 直接查字典 O(1)
    #             slots = solver.edge_available_slots.get((u, v))
    #             if slots:
    #                 return slots
    #     except:
    #         pass

    #     # 2. 如果没有预计算索引，回退到原有的缓存机制
    #     if not hasattr(AlphaDiscreteDistribution, '_slot_cache'):
    #         AlphaDiscreteDistribution._slot_cache = {}
    #     cache_key = (u, v)
    #     if cache_key in AlphaDiscreteDistribution._slot_cache:
    #         return AlphaDiscreteDistribution._slot_cache[cache_key]
        
    #     # 3. 最慢路径：全表扫描 (只在极少数 fallback 情况发生)
    #     available = []
    #     try:
    #         # 尝试访问 link_distributions_backward 或 link_distributions
    #         # 根据你的求解器类定义调整属性名
    #         solver = getattr(get_link_dist_func, '__self__', None)
    #         dists = getattr(solver, 'link_distributions', {}) 
    #         # 或者 getattr(solver, 'link_distributions_backward', {})
            
    #         # 这里其实很难高效获取，因为 link_distributions key 是 (u, v, t)
    #         # 如果没有 edge_available_slots 索引，只能遍历
    #         # 但只要步骤1工作正常，永远不会走到这里
    #         for (link_u, link_v, slot) in dists.keys():
    #             if link_u == u and link_v == v:
    #                 available.append(slot)
    #     except:
    #         pass
        
    #     result = sorted(list(set(available)))
    #     AlphaDiscreteDistribution._slot_cache[cache_key] = result
    #     return result
    
    def _get_slots_in_range(self, slot_min: int, slot_max: int,
                           available_slots: List[int],
                           time_intervals_per_day: int) -> List[int]:
        """获取范围[slot_min, slot_max]内的候选时间片"""
        candidate_slots = []
        
        if slot_min <= slot_max:
            # 正常范围
            for slot in available_slots:
                if slot_min <= slot <= slot_max:
                    candidate_slots.append(slot)
        else:
            # 跨天范围
            for slot in available_slots:
                if slot >= slot_min or slot <= slot_max:
                    candidate_slots.append(slot)
        
        return sorted(candidate_slots)


@dataclass
class LinkTimeDistribution:
    """路段旅行时间分布"""
    time_prob: Dict[int, float]
    times: List[int]
    cdf: List[float]
    time_slot: int
    
    def __init__(self, time_prob_dict: Dict[int, float], time_slot: int = None):
        if not time_prob_dict:
            raise ValueError("链路分布不能为空")
        
        total_prob = sum(time_prob_dict.values())
        self.time_prob = {t: p/total_prob for t, p in time_prob_dict.items()}
        self.time_slot = time_slot
        
        sorted_times = sorted(self.time_prob.keys())
        self.times = sorted_times
        
        cumulative = 0.0
        self.cdf = []
        for t in sorted_times:
            cumulative += self.time_prob[t]
            self.cdf.append(cumulative)
    
    def sample_L2_times(self, reference_time: int, L2: int) -> List[int]:
        """采样L2个旅行时间（逆CDF方法）"""
        samples = []
        for i in range(1, L2 + 1):
            quantile = i / (L2 + 1)
            sample = self._inverse_cdf(quantile)
            samples.append(sample)
        return sorted(samples)
    
    def _inverse_cdf(self, quantile: float) -> int:
        """逆CDF（线性插值）"""
        if quantile <= 0:
            return self.times[0]
        if quantile >= 1:
            return self.times[-1]
        
        for i, cdf_val in enumerate(self.cdf):
            if cdf_val >= quantile:
                if i == 0:
                    return self.times[0]
                
                lower_cdf = self.cdf[i-1] if i > 0 else 0
                upper_cdf = cdf_val
                lower_time = self.times[i-1] if i > 0 else self.times[0]
                upper_time = self.times[i]
                
                if upper_cdf > lower_cdf:
                    weight = (quantile - lower_cdf) / (upper_cdf - lower_cdf)
                else:
                    weight = 0.5
                
                return int(round(lower_time + weight * (upper_time - lower_time)))
        
        return self.times[-1]
    
    def get_probabilities_batch(self, travel_times: np.ndarray) -> np.ndarray:
        """
        批量获取旅行时间的概率（向量化插值）
        
        Args:
            travel_times: 旅行时间数组 (np.ndarray)
        Returns:
            对应概率数组 (np.ndarray)
        """
        # 如果没有定义支撑点，返回全 0
        if not hasattr(self, 'times') or not self.times:
            return np.zeros_like(travel_times, dtype=float)
            
        # 使用 numpy 线性插值
        # self.times 必须是升序排列（你在预处理时已经 sorted 了）
        # left=0.0, right=0.0 表示超出范围返回 0
        return np.interp(
            travel_times, 
            self.times, 
            [self.time_prob[t] for t in self.times], 
            left=0.0, 
            right=0.0
        )
    
    def get_probability(self, travel_time: float) -> float:
        """
        获取指定旅行时间的概率
        
        Args:
            travel_time: 旅行时间（0.1分钟单位）
            
        Returns:
            该旅行时间的概率（如果不在支持集中，可选择插值或返回0）
        """
        # 转换为整数（与存储的键匹配）
        travel_time_int = int(round(travel_time))
        
        # 精确匹配
        if travel_time_int in self.time_prob:
            return self.time_prob[travel_time_int]
        
        # 可选：线性插值（如果需要更平滑的结果）
        if self.times:
            min_time = self.times[0]
            max_time = self.times[-1]
            
            if travel_time_int < min_time or travel_time_int > max_time:
                return 0.0
            
            # 找到相邻的两个点
            for i in range(len(self.times) - 1):
                if self.times[i] <= travel_time_int <= self.times[i+1]:
                    # 线性插值
                    t_lower = self.times[i]
                    t_upper = self.times[i+1]
                    p_lower = self.time_prob[t_lower]
                    p_upper = self.time_prob[t_upper]
                    
                    if t_upper == t_lower:
                        return p_lower
                    
                    weight = (travel_time_int - t_lower) / (t_upper - t_lower)
                    return p_lower * (1 - weight) + p_upper * weight
        
        return 0.0

    def get_mean(self) -> float:
        return sum(t * p for t, p in self.time_prob.items())
    
    def get_std(self) -> float:
        mean = self.get_mean()
        variance = sum(p * (t - mean)**2 for t, p in self.time_prob.items())
        return np.sqrt(variance)


@dataclass
class ReverseLabel:
    """反向搜索标签"""
    node_id: int
    distribution: 'AlphaDiscreteDistribution'
    path: List[int]
    cost: float
    quantile_cache: Dict[float, float] = field(default_factory=dict)
    
    # ✅ 添加缓存属性（在 __post_init__ 中设置）
    mean_cache: float = field(default=0.0, init=False, repr=False)
    variance_cache: float = field(default=0.0, init=False, repr=False)
    std_cache: float = field(default=0.0, init=False, repr=False)
    
    def __post_init__(self):
        """后初始化：预计算常用统计量"""
        # 预计算常用分位数
        for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            self.quantile_cache[q] = self.distribution.get_quantile(q)
        
        # 预计算均值、方差和标准差
        self.mean_cache = self.distribution.get_mean()
        self.variance_cache = self.distribution.get_variance()
        self.std_cache = self.distribution.get_std()
    
    def __lt__(self, other:  'ReverseLabel') -> bool:
        """
        用于优先队列排序
        反向求解：cost 越大越优先（最晚出发时间）
        """
        return self.cost > other.cost
    
    @property
    def expected_value(self) -> float:
        """期望值（使用缓存）"""
        return self.mean_cache
    
    @property
    def std_value(self) -> float:
        """标准差（使用缓存）"""
        return self.std_cache
    
    @property
    def variance_value(self) -> float:
        """方差（使用缓存）"""
        return self.variance_cache
    
    def get_cached_quantile(self, alpha:  float) -> float:
        """
        获取缓存的分位数
        
        Args: 
            alpha: 分位数 (0-1)
            
        Returns: 
            float: 分位数值
        """
        if alpha in self.quantile_cache:
            return self.quantile_cache[alpha]
        
        # 计算并缓存
        value = self.distribution.get_quantile(alpha)
        self.quantile_cache[alpha] = value
        return value
    
    def get_quantile(self, alpha: float) -> float:
        """
        获取分位数（外部接口）
        
        Args:
            alpha: 分位数 (0-1)
            
        Returns:
            float: 分位数值
        """
        return self.get_cached_quantile(alpha)
    
    def dominates_weak(self, other:  'ReverseLabel', alpha: float, epsilon: float = 1e-6) -> bool:
        """
        严格的多目标支配规则（MOSP）
        
        在反向求解中，优化目标为：
        1.主目标：max Q_α (α分位数，最晚出发时间)
        2.次要目标1：max μ (期望出发时间)
        3.次要目标2：min σ² (方差，稳定性)
        
        支配条件：
        self 严格支配 other，当且仅当：
        - Q_α(self) > Q_α(other)  (主目标严格更优)
        - μ(self) >= μ(other)      (次要目标1不劣)
        - σ²(self) <= σ²(other)    (次要目标2不劣)
        
        或者：
        - Q_α(self) >= Q_α(other)  (主目标不劣)
        - 至少一个次要目标严格更优
        - 其他次要目标不劣
        
        Args:
            other:  另一个标签
            alpha: 可靠性参数
            epsilon: 数值容差
            
        Returns:
            bool: 是否支配
        """
        if self.node_id != other.node_id:
            return False
        
        # 获取主目标：α分位数（最晚出发时间）
        # 注意：反向CDF中，我们要找的是 1-α 分位数
        Q_self = self.get_cached_quantile(1 - alpha)
        Q_other = other.get_cached_quantile(1 - alpha)
        
        # 获取次要目标1：期望值
        mu_self = self.expected_value
        mu_other = other.expected_value
        
        # 获取次要目标2：方差
        sigma2_self = self.variance_value
        sigma2_other = other.variance_value
        
        # 策略1：主目标严格更优 + 次要目标不劣
        if Q_self > Q_other + epsilon:   # 主目标严格更优（更晚出发）
            # 检查次要目标是否不劣
            mu_ok = mu_self >= mu_other - epsilon       # 期望不劣
            sigma_ok = sigma2_self <= sigma2_other + epsilon  # 方差不劣
            
            if mu_ok and sigma_ok: 
                return True
        
        # 策略2：主目标不劣 + 至少一个次要目标严格更优
        if Q_self >= Q_other - epsilon:  # 主目标不劣
            # 检查是否有次要目标严格更优
            mu_better = mu_self > mu_other + epsilon
            sigma_better = sigma2_self < sigma2_other - epsilon
            
            # 至少一个次要目标严格更优
            if mu_better or sigma_better:
                # 检查其他目标是否不劣
                mu_ok = mu_self >= mu_other - epsilon
                sigma_ok = sigma2_self <= sigma2_other + epsilon
                
                if mu_ok and sigma_ok:
                    return True
        
        return False
    
    def dominates(self, other:  'ReverseLabel', alpha: float, epsilon: float = 1e-6) -> bool:
        """
        统一的支配检查接口
        
        Args:
            other: 另一个标签
            alpha:  可靠性参数
            epsilon: 数值容差
            
        Returns:
            bool: 是否支配
        """
        return self.dominates_weak(other, alpha, epsilon)
    
    def __eq__(self, other: object) -> bool:
        """相等性判断"""
        if not isinstance(other, ReverseLabel):
            return False
        return (self.node_id == other.node_id and 
                abs(self.cost - other.cost) < 1e-9)
    
    def __hash__(self) -> int:
        """哈希值（用于集合和字典）"""
        return hash((self.node_id, round(self.cost, 6)))
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"ReverseLabel(node={self.node_id}, "
                f"cost={self.cost:.2f}, "
                f"E={self.expected_value:.2f}, "
                f"σ={self.std_value:.2f}, "
                f"path_len={len(self.path)})")
    
    def copy(self) -> 'ReverseLabel':
        """
        创建标签的副本
        
        Returns:
            ReverseLabel: 新的标签副本
        """
        return ReverseLabel(
            node_id=self.node_id,
            distribution=self.distribution,  # 分布对象可以共享
            path=self.path.copy(),  # 路径需要复制
            cost=self.cost
        )
    
    def get_statistics(self) -> dict:
        """
        获取标签的统计信息
        
        Returns:
            dict:  统计信息字典
        """
        return {
            'node_id': self.node_id,
            'cost':  self.cost,
            'expected':  self.expected_value,
            'std': self.std_value,
            'variance': self.variance_value,
            'median': self.get_cached_quantile(0.5),
            'q05': self.get_cached_quantile(0.05),
            'q95': self.get_cached_quantile(0.95),
            'path_length': len(self.path),
            'path': self.path
        }



# ═══════════════════════════════════════════════════════════════════
# 反向求解器主类
# ═══════════════════════════════════════════════════════════════════

class ReverseLabelSettingSolver:
    """反向Label-Setting求解器（完整版）"""
    
    def __init__(self, G, sparse_data, node_to_index, scenario_dates,
                 scenario_probs, time_intervals_per_day,
                 L1: int = 50, L2: int = 10,K = 10,
                 verbose: bool = False,
                 max_labels_per_node: int = 20,
             adj_list=None,  # ✨ 新增
             reverse_adj_list=None,
             link_distributions=None,
             edge_travel_time_bounds=None):   # ✨ 新增
        """初始化"""
        self.G = G
        self.sparse_data = sparse_data
        self.node_to_index = node_to_index
        self.index_to_node = {v: k for k, v in node_to_index.items()}
        self.scenario_dates = scenario_dates
        self.scenario_probs = scenario_probs
        self.time_intervals_per_day = time_intervals_per_day
        self.n_scenarios = len(scenario_dates)
        
        self.L1 = L1
        self.L2 = L2
        self.verbose = verbose

        self.max_labels_per_node = max_labels_per_node
        
        print(f"\n{'='*70}")
        print(f"初始化反向Label-Setting求解器（完整版）")
        print(f"{'='*70}")
        print(f"  算法: 反向Label-Setting with 概率权重")
        print(f"  问题: 预留时间预算")
        print(f"  参数: L1={L1}, L2={L2}")
        print(f"  详细输出: {'开启' if verbose else '关闭'}")
        
        # 构建邻接表
        self.adj_list = defaultdict(list)
        self.reverse_adj_list = defaultdict(list) 
        # 预计算链路分布
        self.link_distributions = {}
        # ✨ 初始化索引
        self.edge_travel_time_bounds = edge_travel_time_bounds if edge_travel_time_bounds is not None else defaultdict(list)


        # 如果传入了预计算数据，直接使用
        if adj_list is not None and reverse_adj_list is not None:
            self.adj_list = adj_list
            self.reverse_adj_list=reverse_adj_list
            print("  ✓ 使用预计算邻接表")
        else:
            # 否则自己构建
            self._build_adjacency_lists()
        
        if link_distributions is not None: 
            self.link_distributions = link_distributions
            print("  ✓ 使用预计算链路分布")
            
        else:
            # 否则自己计算
            self._precompute_link_distributions()

        
        # 统计信息
        self.stats = defaultdict(int)
        self.origin_labels_history = []
        
        print(f"\n✓ 初始化完成")
        print(f"{'='*70}\n")
    
    def _build_adjacency_lists(self):
        """构建邻接表"""
        print(f"  [1/2] 构建邻接表...")
        start_time = time.time()
        
        edges_set = set()
        for (scenario_idx, time_idx, from_idx, to_idx) in self.sparse_data.keys():
            if scenario_idx < self.n_scenarios:
                from_node = self.index_to_node[from_idx]
                to_node = self.index_to_node[to_idx]
                edges_set.add((from_node, to_node))
        
        for from_node, to_node in edges_set:
            self.adj_list[from_node].append(to_node)
            self.reverse_adj_list[to_node].append(from_node)
        
        elapsed = time.time() - start_time
        print(f"      ✓ 完成 (用时 {elapsed:.2f}s) - {len(edges_set):,} 条边")
    
    def _precompute_link_distributions(self):
        """预计算链路分布"""
        print(f"  [2/2] 预计算链路分布...")
        start_time = time.time()
        
        link_time_data = defaultdict(list)
        
        for (scenario_idx, time_idx, from_idx, to_idx), travel_time_minutes in self.sparse_data.items():
            if scenario_idx >= self.n_scenarios:
                continue
            
            from_node = self.index_to_node[from_idx]
            to_node = self.index_to_node[to_idx]
            travel_time_01min = int(travel_time_minutes * 10)
            
            link_time_data[(from_node, to_node, time_idx)].append(travel_time_01min)
        
        distribution_count = 0
        for (u, v, t), times in link_time_data.items():
            time_counts = defaultdict(int)
            for time_val in times:
                time_counts[time_val] += 1
            
            total = len(times)
            time_prob = {time_val: count/total for time_val, count in time_counts.items()}
            
            try:
                self.link_distributions[(u, v, t)] = LinkTimeDistribution(time_prob, time_slot=t)
                distribution_count += 1
            except ValueError:
                continue
        
        elapsed = time.time() - start_time
        print(f"      ✓ 完成 (用时 {elapsed:.2f}s) - {distribution_count:,} 个分布")
    
    def _get_link_distribution_at_slot(self, u: int, v: int, slot: int) -> Optional[LinkTimeDistribution]:
        """获取指定出发时间片的链路分布"""
        if (u, v, slot) in self.link_distributions:
            return self.link_distributions[(u, v, slot)]
        
        # 容差匹配
        tolerance = 5
        candidates = []
        
        for (link_u, link_v, link_t) in self.link_distributions.keys():
            if link_u == u and link_v == v:
                diff = abs(link_t - slot)
                cyclic_diff = min(diff, self.time_intervals_per_day - diff)
                
                if cyclic_diff <= tolerance:
                    candidates.append((link_t, cyclic_diff))
        
        if candidates:
            best_slot = min(candidates, key=lambda x: x[1])[0]
            return self.link_distributions[(u, v, best_slot)]
        
        return None
    



    def solve_k_paths(self, origin: int, destination: int, target_arrival_time: int,
                     alpha: float, K: int = 10, max_labels:  int = 100000,
                     print_interval: int = 100) -> Dict: 
        """
        K-Paths 反向求解
        
        Args: 
            origin: 起点
            destination: 终点
            target_arrival_time: 目标到达时间
            alpha: 可靠性参数
            K: 候选路径数量
            max_labels: 最大标签数
            print_interval: 打印间隔
        
        Returns:
            包含K条候选路径的结果字典
        """
        
        print(f"\n{'='*70}")
        print(f"反向Label-Setting求解（K-Paths版本）")
        print(f"{'='*70}")
        print(f"  起点: {origin}")
        print(f"  终点: {destination}")
        print(f"    目标到达: {self._time_to_string(target_arrival_time)}")
        print(f"  可靠性: α={alpha*100:.1f}%")
        print(f"  候选路径数:  K={K}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # 初始化
        open_labels = []
        node_labels = defaultdict(list)
        origin_candidates = []  # ✅ 存储所有到达起点的候选标签
        
        # 初始标签
        init_dist = AlphaDiscreteDistribution([target_arrival_time] * self.L1, self.L1)
        init_label = ReverseLabel(destination, init_dist, [destination], target_arrival_time)
        
        heapq.heappush(open_labels, init_label)
        node_labels[destination].append(init_label)
        self.stats = defaultdict(int)
        self.stats['labels_generated'] = 1
        
        print(f"开始搜索 K={K} 条候选路径...\n")
        
        iteration = 0
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 主循环：找到K条到达起点的路径
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        while open_labels and (self.stats['labels_generated'] < max_labels or not origin_candidates): 
            iteration += 1
            current_label = heapq.heappop(open_labels)
            
            if self.verbose and (iteration % print_interval == 0 or iteration <= 5):
                print(f"  迭代#{iteration}: 节点{current_label.node_id}, "
                      f"cost={current_label.cost/10:.1f}分, "
                      f"候选数={len(origin_candidates)}")
            
            # ✅ 到达起点：保存为候选路径
            if current_label.node_id == origin:
                latest_departure = current_label.distribution.get_quantile(1 - alpha)
                expected_departure = current_label.mean_cache
                
                # 保存候选路径信息
                candidate_info = {
                    'iteration': iteration,
                    'path': list(reversed(current_label.path)),
                    'distribution': current_label.distribution,
                    'latest_departure': latest_departure,
                    'expected_departure': expected_departure,
                    'median_departure': current_label.distribution.get_median(),
                    'std_departure': np.sqrt(current_label.variance_cache),
                    'variance': current_label.variance_cache,
                    'label': current_label,
                    'alpha': alpha,
                    'rank': None,
                    'is_best':  False  # ← 添加这个字段
                }
                
                origin_candidates.append(candidate_info)
                
                print(f"  🎯 找到候选路径#{len(origin_candidates)}  迭代#{iteration}, "
                      f"Q_{{1-α}}={latest_departure/10:.1f}分, "
                      f"Mean={expected_departure/10:.1f}分, "
                      f"路径长度={len(current_label.path)}")
                
                # ✅ 找到K条路径后继续搜索（确保探索充分）
                if len(origin_candidates) >= K:
                    print(f"\n  ✓ 已找到 {len(origin_candidates)} 条候选路径，停止搜索\n")
                    break
                
                # 继续搜索其他路径
                continue
            
            # 支配性检查（较宽松，保留多样性）
            if self._is_dominated(current_label, node_labels[current_label.node_id], alpha):
                self.stats['labels_dominated'] += 1
                continue
            
            self.stats['labels_extended'] += 1
            
            # 反向扩展
            if current_label.node_id not in self.reverse_adj_list:
                continue
            
            for predecessor in self.reverse_adj_list[current_label.node_id]:
                if predecessor in current_label.path:
                    continue
                
                # 反向卷积
                try:
                    def get_link_dist(u, v, slot):
                        return self._get_link_distribution_at_slot(u, v, slot)
                    
                    get_link_dist.__self__ = self
                    
                    new_dist = current_label.distribution.reverse_convolve(
                        get_link_dist_func=get_link_dist,
                        link_time_ranges=self.edge_travel_time_bounds,
                        predecessor=predecessor,
                        current=current_label.node_id,
                        time_intervals_per_day=self.time_intervals_per_day,
                        L2=self.L2
                    )
                    
                    self.stats['convolutions'] += 1
                    
                except Exception as e:
                    if self.verbose and iteration <= 10:
                        print(f"      ⚠ 卷积失败: {e}")
                    continue
                
                new_cost = new_dist.get_quantile(1 - alpha)
                new_label = ReverseLabel(predecessor, new_dist, 
                                        current_label.path + [predecessor], new_cost)
                
                self.stats['labels_generated'] += 1
                
                # 支配性剪枝
                if self._is_dominated(new_label, node_labels[predecessor], alpha):
                    self.stats['labels_dominated'] += 1
                    continue
                
                # 反向剪枝
                original_count = len(node_labels[predecessor])
                node_labels[predecessor] = [
                    old for old in node_labels[predecessor]
                    if not new_label.dominates_weak(old, alpha)
                ]
                self.stats['labels_dominated'] += (original_count - len(node_labels[predecessor]))
                
                node_labels[predecessor].append(new_label)
                node_labels[predecessor] = self._prune_labels(node_labels[predecessor], alpha)
                heapq.heappush(open_labels, new_label)
            
            # 进度显示
            if not self.verbose and iteration % 100 == 0:
                print(f"  进度: 迭代#{iteration}, 生成{self.stats['labels_generated']: ,}, "
                      f"候选{len(origin_candidates)}, "
                      f"剪枝{self.stats['labels_dominated']:,}", end='\r')
        
        total_time = time.time() - start_time
        
        print(f"\n\n{'='*70}")
        print(f"搜索完成")
        print(f"{'='*70}")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤2：对K条候选路径排序
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        if not origin_candidates:
            print(f"✗ 未找到到达起点的路径")
            return {
                'success': False,
                'total_time': total_time,
                'iterations': iteration,
                'stats':  dict(self.stats),
                'num_candidates': 0
            }
        
        print(f"\n找到 {len(origin_candidates)} 条候选路径")
        print(f"开始排序和比较...\n")
        
        # ✅ 多目标排序：主要Q_{1-α}，次要Mean，再次要-Var
        def rank_score(candidate):
            return (
                candidate['latest_departure'],      # 主目标：Q_{1-α}（越大越好）
                candidate['expected_departure'],    # 次要：均值（越大越好）
                -candidate['variance']              # 再次要：方差（越小越好）
            )
        
        # 排序：从最优到最差
        sorted_candidates = sorted(origin_candidates, key=rank_score, reverse=True)
        
        # 设置排名
        for rank, candidate in enumerate(sorted_candidates, 1):
            candidate['rank'] = rank
            candidate['is_best'] = (rank == 1)  # ← 排名第1的标记为最优
        
        # 取前K条
        top_k_candidates = sorted_candidates[:K]
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 步骤3：输出结果
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        best_candidate = top_k_candidates[0]
        
        print(f"{'='*70}")
        print(f"Top-{len(top_k_candidates)} 候选路径对比")
        print(f"{'='*70}\n")
        
        print(f"{'排名':<6} {'Q_{{1-α}}(分)':<15} {'Mean(分)':<15} {'Std(分)':<12} {'路径长度':<10}")
        print(f"{'-'*70}")
        
        for candidate in top_k_candidates: 
            print(f"{candidate['rank']:<6} "
                  f"{candidate['latest_departure']/10:<15.1f} "
                  f"{candidate['expected_departure']/10:<15.1f} "
                  f"{candidate['std_departure']/10:<12.2f} "
                  f"{len(candidate['path']):<10}")
        
        print(f"\n{'='*70}")
        print(f"✓ 最优路径（排名#1）")
        print(f"{'='*70}")
        print(f"\n  路径:  {self._format_path(best_candidate['path'])}")
        print(f"  长度: {len(best_candidate['path'])} 个节点")
        print(f"\n  时间:")
        print(f"    目标到达: {self._time_to_string(target_arrival_time)}")
        
        print(f"    最晚出发 (α={alpha}): {self._time_to_string(best_candidate['latest_departure'])}")
        print(f"    期望出发: {self._time_to_string(best_candidate['expected_departure'])}")
        print(f"    预留时间: {(target_arrival_time - best_candidate['latest_departure'])/10:.1f}分")
        print(f"    标准差:  {best_candidate['std_departure']/10:.2f}分")
        print(f"\n  性能:")
        print(f"    总耗时: {total_time:.2f}秒")
        print(f"    迭代次数: {iteration}")
        print(f"    候选路径数: {len(origin_candidates)}")
        print(f"    生成标签:  {self.stats['labels_generated']: ,}")
        print(f"    剪枝率: {self.stats['labels_dominated']/self.stats['labels_generated']*100:.1f}%")
        print(f"{'='*70}\n")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 构建返回结果
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        result = {
            'success': True,
            # 最优路径信息
            'path': best_candidate['path'],
            'latest_departure_time': best_candidate['latest_departure'],
            'expected_departure_time': best_candidate['expected_departure'],
            'median_departure_time': best_candidate['median_departure'],
            'std_departure_time': best_candidate['std_departure'],
            'reserved_time': target_arrival_time - best_candidate['latest_departure'],
            'distribution': best_candidate['distribution'],
            
            # Top-K候选路径
            'top_k_candidates': top_k_candidates,
            'num_candidates': len(origin_candidates),
            'all_candidates': sorted_candidates,
            
            # 元信息
            'total_time':  total_time,
            'iterations': iteration,
            'alpha': alpha,
            'K':  K,
            'origin':  origin,
            'destination': destination,
            'target_arrival_time': target_arrival_time,
            'stats': dict(self.stats)
        }
        
        return result

    def solve(self, origin: int, destination: int, target_arrival_time: int,
            alpha: float, max_labels: int = 100000, K:int=5,
            print_interval: int = 100,
            save_all_paths: bool = True) -> Dict: 
        """
        标准solve接口（兼容原代码）
        
        内部调用solve_k_paths，K=5（返回5条候选路径）
        """
        target_arrival_time=target_arrival_time*10  #转为0.1min
        result = self.solve_k_paths(
            origin, destination, target_arrival_time, alpha,
            K=K, max_labels=max_labels, print_interval=print_interval
        )
        
        # 添加all_paths用于兼容
        if save_all_paths and 'all_candidates' in result:
            result['all_paths'] = result['all_candidates']
            result['num_candidate_paths'] = result['num_candidates']
        
        return result
    
    def _is_dominated(self, label: ReverseLabel, existing_labels: List[ReverseLabel], 
                     alpha: float) -> bool:
        """
        支配性检查（保守版本）
        
        策略：
        1.如果节点标签数 < max_labels_per_node：不剪枝
        2.如果已满：检查是否被弱支配
        """
        # 策略1：保留多样性
        if len(existing_labels) < self.max_labels_per_node:
            # 只有被多个标签明确支配时才剪枝
            domination_count = 0
            for existing in existing_labels:
                if existing.dominates_weak(label, alpha):
                    domination_count += 1
            
            # 被2个以上标签支配才剪枝
            return domination_count >= 2
        
        # 策略2：标签数已满，使用弱支配
        for existing in existing_labels:
            if existing.dominates_weak(label, alpha):
                return True
        
        return False
    
    def _prune_labels(self, labels: List[ReverseLabel], alpha: float) -> List[ReverseLabel]:
        """
        当标签数超限时，移除最差的标签
        
        排序标准：
        1.Q_{1-α}（主要）
        2.Mean（次要）
        3.-Var（再次要）
        """
        if len(labels) <= self.max_labels_per_node:
            return labels
        
        # 多目标排序
        def label_score(label):
            q = label.distribution.get_quantile(1 - alpha)
            return (q, label.mean_cache, -label.variance_cache)
        
        # 保留最优的max_labels_per_node个
        sorted_labels = sorted(labels, key=label_score, reverse=True)
        return sorted_labels[:self.max_labels_per_node]

    def _format_path(self, path: List[int]) -> str:
        if len(path) <= 10:
            return ' → '.join(map(str, path))
        return f"{' → '.join(map(str, path[:5]))} → ...→ {' → '.join(map(str, path[-3:]))}"

    def _time_to_string(self, time_01min):
        """时间格式转换"""
        total_minutes = time_01min / 10
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        return f"{hours:02d}:{minutes:02d}"
