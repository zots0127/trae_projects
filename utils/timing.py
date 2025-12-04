#!/usr/bin/env python3
"""
时间统计工具模块
提供统一的时间追踪和性能分析功能
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
from pathlib import Path


@dataclass
class TimeRecord:
    """时间记录"""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def stop(self):
        """停止计时"""
        if self.end_time is None:
            self.end_time = time.perf_counter()
            self.duration = self.end_time - self.start_time
        return self.duration
    
    def get_duration(self) -> float:
        """获取持续时间"""
        if self.duration is not None:
            return self.duration
        elif self.end_time is not None:
            return self.end_time - self.start_time
        else:
            return time.perf_counter() - self.start_time


class TimingTracker:
    """
    时间追踪器
    支持嵌套时间记录和性能统计
    """
    
    def __init__(self, name: str = "main"):
        """
        初始化时间追踪器
        
        Args:
            name: 追踪器名称
        """
        self.name = name
        self.records: Dict[str, TimeRecord] = {}
        self.active_records: List[str] = []
        self.completed_records: List[TimeRecord] = []
        self.start_time = time.perf_counter()
        
    def start(self, name: str, metadata: Optional[Dict] = None) -> TimeRecord:
        """
        开始计时
        
        Args:
            name: 计时名称
            metadata: 附加元数据
        
        Returns:
            时间记录对象
        """
        record = TimeRecord(
            name=name,
            start_time=time.perf_counter(),
            metadata=metadata or {}
        )
        self.records[name] = record
        self.active_records.append(name)
        return record
    
    def stop(self, name: str) -> float:
        """
        停止计时
        
        Args:
            name: 计时名称
        
        Returns:
            持续时间（秒）
        """
        if name in self.records:
            record = self.records[name]
            duration = record.stop()
            
            # 从活动记录中移除
            if name in self.active_records:
                self.active_records.remove(name)
            
            # 添加到完成记录
            self.completed_records.append(record)
            
            return duration
        return 0.0
    
    @contextmanager
    def measure(self, name: str, metadata: Optional[Dict] = None):
        """
        上下文管理器形式的计时
        
        Args:
            name: 计时名称
            metadata: 附加元数据
        
        Example:
            with tracker.measure("feature_extraction"):
                # 执行特征提取
                pass
        """
        record = self.start(name, metadata)
        try:
            yield record
        finally:
            self.stop(name)
    
    def add_metric(self, name: str, key: str, value: Any):
        """
        为时间记录添加指标
        
        Args:
            name: 计时名称
            key: 指标键
            value: 指标值
        """
        if name in self.records:
            self.records[name].metadata[key] = value
    
    def get_duration(self, name: str) -> Optional[float]:
        """
        获取指定记录的持续时间
        
        Args:
            name: 计时名称
        
        Returns:
            持续时间（秒）
        """
        if name in self.records:
            return self.records[name].get_duration()
        return None
    
    def calculate_throughput(self, name: str, samples: int) -> Optional[float]:
        """
        计算吞吐量
        
        Args:
            name: 计时名称
            samples: 样本数量
        
        Returns:
            吞吐量（样本/秒）
        """
        duration = self.get_duration(name)
        if duration and duration > 0:
            throughput = samples / duration
            self.add_metric(name, "samples", samples)
            self.add_metric(name, "throughput", throughput)
            self.add_metric(name, "per_sample_time", duration / samples)
            return throughput
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取时间统计摘要
        
        Returns:
            统计摘要字典
        """
        total_time = time.perf_counter() - self.start_time
        
        summary = {
            "total_time": total_time,
            "records": {}
        }
        
        # 添加所有记录的统计
        for name, record in self.records.items():
            duration = record.get_duration()
            record_data = {
                "duration": duration,
                "percentage": (duration / total_time * 100) if total_time > 0 else 0
            }
            
            # 添加元数据
            if record.metadata:
                record_data["metrics"] = record.metadata
            
            summary["records"][name] = record_data
        
        return summary
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """
        获取详细的时间统计报告
        
        Returns:
            详细报告字典
        """
        summary = self.get_summary()
        
        # 计算关键性能指标
        performance_metrics = {}
        
        # 特征提取性能
        if "feature_extraction" in self.records:
            fe_record = self.records["feature_extraction"]
            if "samples" in fe_record.metadata:
                samples = fe_record.metadata["samples"]
                duration = fe_record.get_duration()
                performance_metrics["feature_extraction"] = {
                    "total_time": duration,
                    "samples": samples,
                    "throughput": f"{samples/duration:.2f} samples/sec" if duration > 0 else "N/A",
                    "per_sample_avg": f"{duration/samples*1000:.2f} ms" if samples > 0 else "N/A"
                }
        
        # 训练性能
        training_records = [name for name in self.records if "training" in name or "fold" in name]
        if training_records:
            total_training_time = sum(self.records[name].get_duration() for name in training_records)
            performance_metrics["training"] = {
                "total_time": total_training_time,
                "phases": len(training_records)
            }
            
            # 分折统计
            fold_times = []
            for name in training_records:
                if "fold" in name:
                    fold_times.append(self.records[name].get_duration())
            
            if fold_times:
                performance_metrics["training"]["fold_times"] = fold_times
                performance_metrics["training"]["avg_fold_time"] = sum(fold_times) / len(fold_times)
        
        # 预测性能
        prediction_records = [name for name in self.records if "predict" in name]
        if prediction_records:
            prediction_metrics = {}
            for name in prediction_records:
                record = self.records[name]
                if "samples" in record.metadata:
                    samples = record.metadata["samples"]
                    duration = record.get_duration()
                    prediction_metrics[name] = {
                        "time": duration,
                        "samples": samples,
                        "throughput": f"{samples/duration:.2f} samples/sec" if duration > 0 else "N/A"
                    }
            performance_metrics["prediction"] = prediction_metrics
        
        return {
            "summary": summary,
            "performance_metrics": performance_metrics,
            "timeline": self._get_timeline()
        }
    
    def _get_timeline(self) -> List[Dict]:
        """
        获取时间线
        
        Returns:
            时间线列表
        """
        timeline = []
        for record in self.completed_records:
            timeline.append({
                "name": record.name,
                "start": record.start_time - self.start_time,
                "end": (record.end_time - self.start_time) if record.end_time else None,
                "duration": record.duration
            })
        
        # 按开始时间排序
        timeline.sort(key=lambda x: x["start"])
        return timeline
    
    def save_report(self, filepath: Path, format: str = "json"):
        """
        保存时间统计报告
        
        Args:
            filepath: 保存路径
            format: 格式（json/txt）
        """
        report = self.get_detailed_report()
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        elif format == "txt":
            with open(filepath, 'w') as f:
                f.write(self._format_text_report(report))
    
    def _format_text_report(self, report: Dict) -> str:
        """
        格式化文本报告
        
        Args:
            report: 报告字典
        
        Returns:
            格式化的文本
        """
        lines = []
        lines.append("="*60)
        lines.append(f"时间统计报告 - {self.name}")
        lines.append("="*60)
        
        summary = report["summary"]
        lines.append(f"\n总耗时: {summary['total_time']:.2f} 秒")
        lines.append("\n各阶段耗时:")
        lines.append("-"*40)
        
        for name, data in summary["records"].items():
            lines.append(f"  {name:30s}: {data['duration']:8.2f}s ({data['percentage']:5.1f}%)")
            if "metrics" in data and "throughput" in data["metrics"]:
                lines.append(f"    吞吐量: {data['metrics']['throughput']:.2f} samples/sec")
        
        if "performance_metrics" in report:
            lines.append("\n性能指标:")
            lines.append("-"*40)
            
            metrics = report["performance_metrics"]
            
            if "feature_extraction" in metrics:
                fe = metrics["feature_extraction"]
                lines.append(f"  特征提取:")
                lines.append(f"    - 总时间: {fe['total_time']:.2f}s")
                lines.append(f"    - 吞吐量: {fe['throughput']}")
                lines.append(f"    - 平均每样本: {fe['per_sample_avg']}")
            
            if "training" in metrics:
                tr = metrics["training"]
                lines.append(f"  模型训练:")
                lines.append(f"    - 总时间: {tr['total_time']:.2f}s")
                if "avg_fold_time" in tr:
                    lines.append(f"    - 平均每折: {tr['avg_fold_time']:.2f}s")
            
            if "prediction" in metrics:
                lines.append(f"  预测:")
                for name, pred in metrics["prediction"].items():
                    lines.append(f"    {name}:")
                    lines.append(f"      - 时间: {pred['time']:.2f}s")
                    lines.append(f"      - 吞吐量: {pred['throughput']}")
        
        lines.append("\n" + "="*60)
        return "\n".join(lines)
    
    def print_summary(self):
        """打印时间统计摘要"""
        report = self.get_detailed_report()
        print(self._format_text_report(report))


# 全局时间追踪器实例
_global_tracker: Optional[TimingTracker] = None


def get_global_tracker() -> TimingTracker:
    """获取全局时间追踪器"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TimingTracker("global")
    return _global_tracker


def reset_global_tracker():
    """重置全局时间追踪器"""
    global _global_tracker
    _global_tracker = TimingTracker("global")


if __name__ == "__main__":
    # 测试代码
    import numpy as np
    
    tracker = TimingTracker("test")
    
    # 测试特征提取计时
    with tracker.measure("feature_extraction", {"phase": "morgan_fingerprints"}):
        time.sleep(0.5)  # 模拟特征提取
        tracker.calculate_throughput("feature_extraction", 1000)
    
    # 测试训练计时
    for i in range(3):
        with tracker.measure(f"fold_{i+1}_training"):
            time.sleep(0.2)  # 模拟训练
    
    # 测试预测计时
    with tracker.measure("test_prediction"):
        time.sleep(0.1)  # 模拟预测
        tracker.calculate_throughput("test_prediction", 500)
    
    # 打印摘要
    tracker.print_summary()
    
    # 保存报告
    tracker.save_report(Path("test_timing_report.json"), format="json")
    tracker.save_report(Path("test_timing_report.txt"), format="txt")
    
    print("\n测试完成！")