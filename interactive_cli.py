#!/usr/bin/env python3
"""
交互式CLI管理界面
提供用户友好的项目管理和批量预测界面
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import pandas as pd
from datetime import datetime
import subprocess
import shlex

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 尝试导入rich，如果没有则使用基础版本
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.layout import Layout
    from rich.text import Text
    from rich import print as rprint
    from rich.columns import Columns
    from rich.tree import Tree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ Rich library not installed. Using basic interface.")
    print("Install with: pip install rich")

from utils.project_manager import ProjectManager
from utils.project_predictor import ProjectPredictor


class InteractiveCLI:
    """交互式CLI管理界面"""
    
    def __init__(self):
        """初始化CLI"""
        self.console = Console() if RICH_AVAILABLE else None
        self.manager = ProjectManager()
        self.current_project = None
        self.current_predictor = None
        
    def clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """打印标题"""
        self.clear_screen()
        if RICH_AVAILABLE:
            header = Panel.fit(
                "[bold cyan]AutoML Interactive Manager[/bold cyan]\n"
                "[dim]Project Management & Batch Prediction[/dim]",
                border_style="cyan"
            )
            self.console.print(header)
        else:
            print("=" * 60)
            print("         AutoML Interactive Manager")
            print("    Project Management & Batch Prediction")
            print("=" * 60)
    
    def main_menu(self) -> str:
        """主菜单"""
        self.print_header()
        
        if self.current_project:
            if RICH_AVAILABLE:
                self.console.print(f"\n📦 Current Project: [bold green]{self.current_project}[/bold green]\n")
            else:
                print(f"\n📦 Current Project: {self.current_project}\n")
        
        menu_items = [
            "1. 📋 List Projects",
            "2. 📂 Select Project",
            "3. 📊 Project Information",
            "4. 🚀 Batch Prediction",
            "5. 🎯 Train New Models",
            "6. 📈 View Comparison Table",
            "7. 💾 Export Project",
            "8. 📊 Generate Comparison Table",
            "9. 📝 Generate Report",
            "10. 🧹 Clean Project",
            "0. 🚪 Exit"
        ]
        
        if RICH_AVAILABLE:
            menu = Panel("\n".join(menu_items), title="Main Menu", border_style="blue")
            self.console.print(menu)
            choice = Prompt.ask("\n[bold]Your choice[/bold]", choices=["0","1","2","3","4","5","6","7","8","9","10"])
        else:
            print("\nMain Menu:")
            for item in menu_items:
                print(f"  {item}")
            choice = input("\nYour choice: ")
        
        return choice
    
    def list_projects(self):
        """列出所有项目"""
        self.print_header()
        projects = self.manager.list_projects()
        
        if not projects:
            if RICH_AVAILABLE:
                self.console.print("[yellow]No projects found.[/yellow]")
            else:
                print("No projects found.")
            return
        
        if RICH_AVAILABLE:
            table = Table(title="Available Projects", show_header=True, header_style="bold magenta")
            table.add_column("Project", style="cyan", no_wrap=True)
            table.add_column("Created", style="green")
            table.add_column("Models", justify="right", style="yellow")
            table.add_column("Runs", justify="right", style="yellow")
            table.add_column("Path", style="dim")
            
            for p in projects:
                table.add_row(
                    p['name'],
                    p['created'][:19] if p['created'] != 'Unknown' else 'Unknown',
                    str(p['models']),
                    str(p['runs']),
                    str(p['path'])
                )
            
            self.console.print(table)
        else:
            print("\nAvailable Projects:")
            print("-" * 60)
            for i, p in enumerate(projects, 1):
                print(f"{i}. {p['name']}")
                print(f"   Created: {p['created']}")
                print(f"   Models: {p['models']}, Runs: {p['runs']}")
                print(f"   Path: {p['path']}")
                print()
    
    def select_project(self):
        """选择项目"""
        self.print_header()
        projects = self.manager.list_projects()
        
        if not projects:
            if RICH_AVAILABLE:
                self.console.print("[yellow]No projects found.[/yellow]")
            else:
                print("No projects found.")
            return
        
        # 显示项目列表
        if RICH_AVAILABLE:
            self.console.print("[bold]Available Projects:[/bold]\n")
            for i, p in enumerate(projects, 1):
                self.console.print(f"  {i}. [cyan]{p['name']}[/cyan] ({p['models']} models)")
            
            choice = IntPrompt.ask("\n[bold]Select project number[/bold]", 
                                  default=1, 
                                  show_default=True)
        else:
            print("\nAvailable Projects:\n")
            for i, p in enumerate(projects, 1):
                print(f"  {i}. {p['name']} ({p['models']} models)")
            
            choice = input("\nSelect project number (default=1): ")
            choice = int(choice) if choice else 1
        
        if 1 <= choice <= len(projects):
            self.current_project = projects[choice - 1]['name']
            self.current_predictor = None  # Reset predictor
            
            if RICH_AVAILABLE:
                self.console.print(f"\n✅ Selected project: [bold green]{self.current_project}[/bold green]")
            else:
                print(f"\n✅ Selected project: {self.current_project}")
        else:
            if RICH_AVAILABLE:
                self.console.print("[red]Invalid selection.[/red]")
            else:
                print("Invalid selection.")
    
    def show_project_info(self):
        """显示项目信息"""
        if not self.current_project:
            if RICH_AVAILABLE:
                self.console.print("[yellow]Please select a project first.[/yellow]")
            else:
                print("Please select a project first.")
            return
        
        self.print_header()
        
        try:
            # 获取项目信息
            info = self.manager.get_project_info(self.current_project)
            
            # 加载预测器获取模型详情
            if not self.current_predictor:
                self.current_predictor = ProjectPredictor(self.current_project, verbose=False)
            
            if RICH_AVAILABLE:
                # 基本信息
                info_panel = Panel(
                    f"[bold]Name:[/bold] {info['project_name']}\n"
                    f"[bold]Created:[/bold] {info.get('created_at', 'Unknown')[:19]}\n"
                    f"[bold]Path:[/bold] {info['path']}\n"
                    f"[bold]Models:[/bold] {len(self.current_predictor.models)}\n"
                    f"[bold]Targets:[/bold] {', '.join(info.get('targets', []))}",
                    title="Project Information",
                    border_style="green"
                )
                self.console.print(info_panel)
                
                # 模型列表
                if self.current_predictor.models:
                    table = Table(title="Trained Models", show_header=True, header_style="bold cyan")
                    table.add_column("Model", style="cyan")
                    table.add_column("Target", style="green")
                    table.add_column("R² (mean±std)", justify="right", style="yellow")
                    table.add_column("RMSE (mean±std)", justify="right", style="yellow")
                    table.add_column("MAE (mean±std)", justify="right", style="yellow")
                    
                    for key, info in self.current_predictor.models.items():
                        perf = info.get('performance', {})
                        
                        # 格式化 R² with std
                        r2_str = 'N/A'
                        if isinstance(perf.get('r2'), float):
                            r2_mean = perf.get('r2')
                            r2_std = perf.get('r2_std', 0)
                            if r2_std > 0:
                                r2_str = f"{r2_mean:.4f}±{r2_std:.4f}"
                            else:
                                r2_str = f"{r2_mean:.4f}"
                        
                        # 格式化 RMSE with std
                        rmse_str = 'N/A'
                        if isinstance(perf.get('rmse'), float):
                            rmse_mean = perf.get('rmse')
                            rmse_std = perf.get('rmse_std', 0)
                            if rmse_std > 0:
                                rmse_str = f"{rmse_mean:.2f}±{rmse_std:.2f}"
                            else:
                                rmse_str = f"{rmse_mean:.2f}"
                        
                        # 格式化 MAE with std
                        mae_str = 'N/A'
                        if isinstance(perf.get('mae'), float):
                            mae_mean = perf.get('mae')
                            mae_std = perf.get('mae_std', 0)
                            if mae_std > 0:
                                mae_str = f"{mae_mean:.2f}±{mae_std:.2f}"
                            else:
                                mae_str = f"{mae_mean:.2f}"
                        
                        table.add_row(
                            info['type'],
                            info.get('original_target', info['target']),
                            r2_str,
                            rmse_str,
                            mae_str
                        )
                    
                    self.console.print(table)
                
                # 最佳模型
                if info.get('best_models'):
                    best_panel = Panel(
                        "\n".join([
                            f"[bold]{target}:[/bold] {best['model']} (R²={best['r2']:.4f})"
                            for target, best in info['best_models'].items()
                        ]),
                        title="Best Models",
                        border_style="yellow"
                    )
                    self.console.print(best_panel)
            else:
                # 基础文本输出
                print("\nProject Information:")
                print("-" * 60)
                print(f"Name: {info['project_name']}")
                print(f"Created: {info.get('created_at', 'Unknown')}")
                print(f"Path: {info['path']}")
                print(f"Models: {len(self.current_predictor.models)}")
                print(f"Targets: {', '.join(info.get('targets', []))}")
                
                print("\nTrained Models:")
                for key, model_info in self.current_predictor.models.items():
                    perf = model_info.get('performance', {})
                    print(f"  - {model_info['type']} on {model_info.get('original_target', model_info['target'])}")
                    if perf:
                        print(f"    R²={perf.get('r2', 'N/A'):.4f}, RMSE={perf.get('rmse', 'N/A'):.4f}")
                
                if info.get('best_models'):
                    print("\nBest Models:")
                    for target, best in info['best_models'].items():
                        print(f"  {target}: {best['model']} (R²={best['r2']:.4f})")
                        
        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"[red]Error: {e}[/red]")
            else:
                print(f"Error: {e}")
    
    def batch_prediction(self):
        """批量预测交互流程"""
        if not self.current_project:
            if RICH_AVAILABLE:
                self.console.print("[yellow]Please select a project first.[/yellow]")
            else:
                print("Please select a project first.")
            return
        
        self.print_header()
        
        # 获取数据文件
        if RICH_AVAILABLE:
            self.console.print("[bold]Batch Prediction Setup[/bold]\n")
            data_file = Prompt.ask("Enter data file path", default="data/Database_normalized.csv")
        else:
            print("\nBatch Prediction Setup\n")
            data_file = input("Enter data file path (default=data/Database_normalized.csv): ")
            data_file = data_file or "data/Database_normalized.csv"
        
        # 检查文件是否存在
        if not Path(data_file).exists():
            if RICH_AVAILABLE:
                self.console.print(f"[red]File not found: {data_file}[/red]")
            else:
                print(f"File not found: {data_file}")
            return
        
        # 选择预测模式
        modes = {
            "1": ("best", "Use best models only"),
            "2": ("all", "Use all models"),
            "3": ("ensemble", "Ensemble prediction")
        }
        
        if RICH_AVAILABLE:
            self.console.print("\n[bold]Prediction Mode:[/bold]")
            for key, (mode, desc) in modes.items():
                self.console.print(f"  {key}. {desc}")
            
            mode_choice = Prompt.ask("Select mode", choices=["1", "2", "3"], default="1")
        else:
            print("\nPrediction Mode:")
            for key, (mode, desc) in modes.items():
                print(f"  {key}. {desc}")
            
            mode_choice = input("Select mode (1/2/3, default=1): ")
            mode_choice = mode_choice or "1"
        
        mode = modes[mode_choice][0]
        
        # 如果是ensemble，询问方法
        method = "mean"
        if mode == "ensemble":
            methods = {"1": "mean", "2": "median", "3": "weighted"}
            if RICH_AVAILABLE:
                self.console.print("\n[bold]Ensemble Method:[/bold]")
                for key, m in methods.items():
                    self.console.print(f"  {key}. {m}")
                method_choice = Prompt.ask("Select method", choices=["1", "2", "3"], default="1")
            else:
                print("\nEnsemble Method:")
                for key, m in methods.items():
                    print(f"  {key}. {m}")
                method_choice = input("Select method (1/2/3, default=1): ")
                method_choice = method_choice or "1"
            method = methods[method_choice]
        
        # 输出文件
        default_output = f"{self.current_project}/predictions_{mode}.csv"
        
        if RICH_AVAILABLE:
            output_file = Prompt.ask("Output file", default=default_output)
        else:
            output_file = input(f"Output file (default={default_output}): ")
            output_file = output_file or default_output
        
        # 确认执行
        if RICH_AVAILABLE:
            self.console.print("\n[bold]Summary:[/bold]")
            self.console.print(f"  Project: {self.current_project}")
            self.console.print(f"  Data: {data_file}")
            self.console.print(f"  Mode: {mode}")
            if mode == "ensemble":
                self.console.print(f"  Method: {method}")
            self.console.print(f"  Output: {output_file}")
            
            if not Confirm.ask("\nProceed with prediction?", default=True):
                return
        else:
            print("\nSummary:")
            print(f"  Project: {self.current_project}")
            print(f"  Data: {data_file}")
            print(f"  Mode: {mode}")
            if mode == "ensemble":
                print(f"  Method: {method}")
            print(f"  Output: {output_file}")
            
            proceed = input("\nProceed with prediction? (y/n, default=y): ")
            if proceed.lower() == 'n':
                return
        
        # 执行预测
        try:
            if RICH_AVAILABLE:
                with self.console.status("[bold green]Running prediction...", spinner="dots"):
                    self._run_prediction(data_file, mode, output_file, method)
            else:
                print("\nRunning prediction...")
                self._run_prediction(data_file, mode, output_file, method)
            
            if RICH_AVAILABLE:
                self.console.print(f"\n✅ [green]Prediction completed![/green]")
                self.console.print(f"   Output saved to: {output_file}")
            else:
                print(f"\n✅ Prediction completed!")
                print(f"   Output saved to: {output_file}")
                
        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"[red]Prediction failed: {e}[/red]")
            else:
                print(f"Prediction failed: {e}")
    
    def _run_prediction(self, data_file: str, mode: str, output_file: str, method: str = "mean"):
        """执行预测"""
        if not self.current_predictor:
            self.current_predictor = ProjectPredictor(self.current_project, verbose=False)
        
        if mode == "best":
            self.current_predictor.predict_best_models(
                data_path=data_file,
                output_path=output_file
            )
        elif mode == "all":
            output_dir = Path(output_file).parent / f"batch_{Path(output_file).stem}"
            self.current_predictor.predict_all_models(
                data_path=data_file,
                output_dir=str(output_dir)
            )
        elif mode == "ensemble":
            self.current_predictor.predict_ensemble(
                data_path=data_file,
                output_path=output_file,
                method=method
            )
    
    def view_comparison_table(self):
        """查看对比表"""
        if not self.current_project:
            if RICH_AVAILABLE:
                self.console.print("[yellow]Please select a project first.[/yellow]")
            else:
                print("Please select a project first.")
            return
        
        self.print_header()
        
        # 查找对比表文件
        project_path = Path(self.current_project)
        table_files = list(project_path.glob("comparison_table_*.csv"))
        
        if not table_files:
            # 若没有现成表格，提示是否立即生成
            if RICH_AVAILABLE:
                self.console.print("[yellow]No comparison tables found. Let's generate one now.[/yellow]")
                if not Confirm.ask("Generate comparison table now?", default=True):
                    return
            else:
                print("No comparison tables found.")
                proceed = input("Generate comparison table now? (y/n, default=y): ")
                if proceed.lower() == 'n':
                    return

            try:
                self.generate_comparison_table(auto_after_view=True)
            except Exception as e:
                if RICH_AVAILABLE:
                    self.console.print(f"[red]Failed to generate table: {e}[/red]")
                else:
                    print(f"Failed to generate table: {e}")
                return

            # 重新加载
            table_files = list(project_path.glob("comparison_table_*.csv"))
            if not table_files:
                if RICH_AVAILABLE:
                    self.console.print("[yellow]Still no tables found after generation.[/yellow]")
                else:
                    print("Still no tables found after generation.")
                return
        
        # 读取最新的表格
        latest_table = sorted(table_files)[-1]
        df = pd.read_csv(latest_table)
        
        if RICH_AVAILABLE:
            self.console.print(f"[bold]Comparison Table:[/bold] {latest_table.name}\n")
            
            # 转换为rich表格
            table = Table(show_header=True, header_style="bold magenta", title=f"Model Comparison Table")
            
            # 智能添加列 - 识别mean和std列
            col_groups = {}
            for col in df.columns:
                if '_mean' in col:
                    base_name = col.replace('_mean', '')
                    if base_name not in col_groups:
                        col_groups[base_name] = {}
                    col_groups[base_name]['mean'] = col
                elif '_std' in col:
                    base_name = col.replace('_std', '')
                    if base_name not in col_groups:
                        col_groups[base_name] = {}
                    col_groups[base_name]['std'] = col
            
            # 添加列头
            for col in df.columns:
                # 检查是否是成对的mean/std列
                is_metric_col = False
                for base_name, group in col_groups.items():
                    if col == group.get('mean'):
                        # 这是mean列，检查是否有对应的std
                        if 'std' in group:
                            table.add_column(f"{base_name} (mean±std)", justify="right", style="yellow")
                        else:
                            table.add_column(col, justify="right", style="yellow")
                        is_metric_col = True
                        break
                    elif col == group.get('std'):
                        # std列会和mean列合并，跳过
                        is_metric_col = True
                        break
                
                if not is_metric_col:
                    # 普通列
                    if 'R2' in col or 'RMSE' in col or 'MAE' in col:
                        table.add_column(col, justify="right", style="yellow")
                    else:
                        table.add_column(col, style="cyan")
            
            # 添加行
            for _, row in df.iterrows():
                row_data = []
                processed_cols = set()
                
                for col in df.columns:
                    if col in processed_cols:
                        continue
                    
                    # 检查是否需要合并mean和std
                    merged = False
                    for base_name, group in col_groups.items():
                        if col == group.get('mean') and 'std' in group:
                            mean_val = row[col]
                            std_val = row[group['std']]
                            if isinstance(mean_val, float) and isinstance(std_val, float):
                                row_data.append(f"{mean_val:.4f}±{std_val:.4f}")
                            else:
                                row_data.append(str(mean_val))
                            processed_cols.add(col)
                            processed_cols.add(group['std'])
                            merged = True
                            break
                    
                    if not merged and col not in processed_cols:
                        # 检查是否是单独的std列（应该已被处理）
                        is_std_col = any(col == g.get('std') for g in col_groups.values())
                        if not is_std_col:
                            val = row[col]
                            if isinstance(val, float):
                                row_data.append(f"{val:.4f}")
                            else:
                                row_data.append(str(val))
                            processed_cols.add(col)
                
                table.add_row(*row_data)
            
            self.console.print(table)
        else:
            print(f"\nComparison Table: {latest_table.name}\n")
            print(df.to_string(index=False))

    def generate_comparison_table(self, auto_after_view: bool = False):
        """生成模型对比表（整合到Manager）"""
        if not self.current_project:
            if RICH_AVAILABLE:
                self.console.print("[yellow]Please select a project first.[/yellow]")
            else:
                print("Please select a project first.")
            return

        self.print_header()

        output_dir = None
        formats = ['markdown', 'html', 'latex', 'csv']

        if not auto_after_view:
            if RICH_AVAILABLE:
                self.console.print("[bold]Generate Comparison Table[/bold]\n")
                output_dir = Prompt.ask("Output directory (default=project root)", default="")
                fmt_choices = ["markdown","html","latex","csv"]
                fmt_input = Prompt.ask("Formats (comma separated)", default=",".join(fmt_choices))
                formats = [f.strip() for f in fmt_input.split(',') if f.strip()]
            else:
                print("Generate Comparison Table\n")
                output_dir = input("Output directory (default=project root): ")
                fmt_input = input("Formats (comma separated, default=markdown,html,latex,csv): ")
                if fmt_input:
                    formats = [f.strip() for f in fmt_input.split(',') if f.strip()]

        try:
            exported = self.manager.generate_comparison_table(
                self.current_project,
                output_dir=output_dir or None,
                formats=formats
            )

            if RICH_AVAILABLE:
                self.console.print("\n✅ [green]Comparison table generated.[/green]")
                for k, v in exported.items():
                    self.console.print(f"   - {k}: {v}")
                # 生成后在命令行内渲染展示一次
                try:
                    self.console.print("\n[bold]Preview (rendered in console):[/bold]\n")
                    self.view_comparison_table()
                except Exception:
                    pass
            else:
                print("\n✅ Comparison table generated.")
                for k, v in exported.items():
                    print(f"   - {k}: {v}")
                # 生成后在命令行内渲染展示一次
                try:
                    print("\nPreview (rendered in console):\n")
                    self.view_comparison_table()
                except Exception:
                    pass
        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"[red]Generation failed: {e}[/red]")
            else:
                print(f"Generation failed: {e}")
    
    def export_project(self):
        """导出项目"""
        if not self.current_project:
            if RICH_AVAILABLE:
                self.console.print("[yellow]Please select a project first.[/yellow]")
            else:
                print("Please select a project first.")
            return
        
        self.print_header()
        
        # 选择格式
        if RICH_AVAILABLE:
            format_choice = Prompt.ask("Export format", choices=["zip", "tar"], default="zip")
            default_output = f"{self.current_project}.{format_choice}"
            output_file = Prompt.ask("Output file", default=default_output)
        else:
            format_choice = input("Export format (zip/tar, default=zip): ")
            format_choice = format_choice or "zip"
            default_output = f"{self.current_project}.{format_choice}"
            output_file = input(f"Output file (default={default_output}): ")
            output_file = output_file or default_output
        
        try:
            self.manager.export_project(self.current_project, output_file, format_choice)
            
            if RICH_AVAILABLE:
                self.console.print(f"✅ [green]Project exported to: {output_file}[/green]")
            else:
                print(f"✅ Project exported to: {output_file}")
                
        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"[red]Export failed: {e}[/red]")
            else:
                print(f"Export failed: {e}")
    
    def generate_report(self):
        """生成报告"""
        if not self.current_project:
            if RICH_AVAILABLE:
                self.console.print("[yellow]Please select a project first.[/yellow]")
            else:
                print("Please select a project first.")
            return
        
        self.print_header()
        
        default_output = f"{self.current_project}/report.md"
        
        if RICH_AVAILABLE:
            output_file = Prompt.ask("Report file", default=default_output)
        else:
            output_file = input(f"Report file (default={default_output}): ")
            output_file = output_file or default_output
        
        try:
            self.manager.generate_project_report(self.current_project, output_file)
            
            if RICH_AVAILABLE:
                self.console.print(f"✅ [green]Report generated: {output_file}[/green]")
            else:
                print(f"✅ Report generated: {output_file}")
                
        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"[red]Report generation failed: {e}[/red]")
            else:
                print(f"Report generation failed: {e}")
    
    def train_models(self):
        """训练新模型"""
        self.print_header()
        
        if RICH_AVAILABLE:
            self.console.print("[bold]Train New Models[/bold]\n")
            self.console.print("This will launch the training pipeline.\n")
            
            # 配置编号选择 - 增加模型信息显示和自定义选项
            configs = ["xgboost_quick", "xgboost_standard", "automl_quick", "automl", "paper_comparison", "custom"]
            
            # 每个配置支持的模型
            config_models = {
                "xgboost_quick": ["XGBoost"],
                "xgboost_standard": ["XGBoost"],
                "automl_quick": ["XGBoost", "LightGBM", "CatBoost", "Random Forest"],
                "automl": ["XGBoost", "LightGBM", "CatBoost", "Random Forest", "Gradient Boosting", 
                          "Extra Trees", "AdaBoost", "Ridge", "Lasso", "Elastic Net", "SVR", "KNN", "Decision Tree"],
                "paper_comparison": ["XGBoost", "LightGBM", "CatBoost", "Random Forest", "Gradient Boosting", 
                                   "Extra Trees", "AdaBoost", "Ridge", "Lasso", "Elastic Net", "SVR", "KNN", "Decision Tree"],
                "custom": []  # Will be filled by user selection
            }
            
            self.console.print("[bold]Select configuration:[/bold]")
            for i, c in enumerate(configs, 1):
                if c == "custom":
                    self.console.print(f"  {i}. [cyan]{c}[/cyan] [dim](Select individual models)[/dim]")
                else:
                    models = config_models.get(c, [])
                    models_str = f"[dim]({len(models)} models: {', '.join(models[:3])}{', ...' if len(models) > 3 else ''})[/dim]"
                    self.console.print(f"  {i}. [cyan]{c}[/cyan] {models_str}")
            
            # 显示详细的模型列表
            self.console.print("\n[bold]Supported models by configuration:[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Config", style="cyan")
            table.add_column("Models", style="yellow", overflow="fold")
            table.add_column("Count", justify="center", style="green")
            
            for config in configs:
                models = config_models.get(config, [])
                # 显示所有模型，不截断
                models_display = ", ".join(models)
                table.add_row(config, models_display, str(len(models)))
            
            self.console.print(table)
            
            # 额外显示所有13个可用模型的完整列表
            self.console.print("\n[bold]All Available Models (13 total):[/bold]")
            all_models = [
                ("XGBoost", "🚀 Tree-based ensemble"),
                ("LightGBM", "🚀 Tree-based ensemble"),
                ("CatBoost", "🚀 Tree-based ensemble"),
                ("Random Forest", "🌲 Tree-based ensemble"),
                ("Gradient Boosting", "🌲 Tree-based ensemble"),
                ("Extra Trees", "🌲 Tree-based ensemble"),
                ("AdaBoost", "🌲 Tree-based ensemble"),
                ("Decision Tree", "🌳 Single tree"),
                ("Ridge", "📊 Linear model"),
                ("Lasso", "📊 Linear model"),
                ("Elastic Net", "📊 Linear model"),
                ("SVR", "🔮 Support Vector"),
                ("KNN", "📍 Instance-based")
            ]
            
            # 创建一个表格显示所有模型
            model_table = Table(show_header=True, header_style="bold cyan", box=None)
            model_table.add_column("#", justify="right", style="dim")
            model_table.add_column("Model", style="yellow")
            model_table.add_column("Type", style="dim")
            
            for i, (model, model_type) in enumerate(all_models, 1):
                model_table.add_row(str(i), model, model_type)
            
            self.console.print(model_table)
            self.console.print()
            
            config_idx = IntPrompt.ask("Configuration number", default=1, show_default=True)
            config_idx = 1 if not isinstance(config_idx, int) else max(1, min(config_idx, len(configs)))
            config = configs[config_idx - 1]
            
            # 如果选择了custom，让用户选择具体的模型
            custom_models_list = None
            custom_models_selected = False  # 标记是否选择了自定义模型
            if config == "custom":
                self.console.print("\n[bold]Select models to train:[/bold]")
                self.console.print("[dim]Enter model numbers separated by commas (e.g., 1,2,3 or 1-5,7,9)[/dim]\n")
                
                # 显示所有可选模型（注意顺序要与标准模式一致）
                all_model_names = ["adaboost", "catboost", "decision_tree", "elastic_net", 
                                  "extra_trees", "gradient_boosting", "knn", "lasso", 
                                  "lightgbm", "random_forest", "ridge", "svr", "xgboost"]
                all_model_display = ["AdaBoost", "CatBoost", "Decision Tree", "Elastic Net", 
                                    "Extra Trees", "Gradient Boosting", "KNN", "Lasso", 
                                    "LightGBM", "Random Forest", "Ridge", "SVR", "XGBoost"]
                
                # 创建模型选择表格
                for i, (name, display) in enumerate(zip(all_model_names, all_model_display), 1):
                    emoji = "🚀" if name in ["xgboost", "lightgbm", "catboost"] else \
                            "🌲" if name in ["random_forest", "gradient_boosting", "extra_trees", "adaboost"] else \
                            "🌳" if name == "decision_tree" else \
                            "📊" if name in ["ridge", "lasso", "elastic_net"] else \
                            "🔮" if name == "svr" else "📍"
                    self.console.print(f"  {i:2}. {emoji} [yellow]{display:20}[/yellow] [dim]({name})[/dim]")
                
                # 获取用户选择
                selection = Prompt.ask("\nSelect models", default="1,2,3,4")
                
                # 解析选择（支持范围如1-5和单个数字）
                selected_indices = []
                for part in selection.split(','):
                    part = part.strip()
                    if '-' in part:
                        try:
                            start, end = map(int, part.split('-'))
                            selected_indices.extend(range(start, end + 1))
                        except:
                            pass
                    else:
                        try:
                            selected_indices.append(int(part))
                        except:
                            pass
                
                # 转换为模型名称
                selected_models = [all_model_names[i-1] for i in selected_indices 
                                 if 1 <= i <= len(all_model_names)]
                
                if not selected_models:
                    selected_models = ["xgboost"]  # 默认至少选择XGBoost
                
                self.console.print(f"\n[green]Selected {len(selected_models)} models:[/green] {', '.join([all_model_display[all_model_names.index(m)] for m in selected_models])}\n")
                
                # 设置为automl模式但只使用选定的模型
                config = "automl"
                # 保存选定的模型列表，稍后在构建命令时处理
                custom_models_list = selected_models
                custom_models_selected = True  # 标记已选择自定义模型
            
            # 获取数据文件
            data_file = Prompt.ask("Data file", default="data/Database_normalized.csv")
            
            # 获取项目名称
            project = Prompt.ask("Project name", default="project")
            
            # 交互式附加选项（编号选择）
            extra_args = []
            if custom_models_list:
                # 使用新的models参数格式（逗号分隔）
                models_str = ','.join(custom_models_list)
                extra_args.append(f"models={models_str}")

            # 特征类型
            feat_options = ["auto", "combined", "morgan", "descriptors", "tabular"]
            self.console.print("\n[bold]Feature type:[/bold]")
            for i, fopt in enumerate(feat_options, 1):
                self.console.print(f"  {i}. {fopt}")
            feat_idx = IntPrompt.ask("Feature type number", default=2)
            feat_idx = 2 if not isinstance(feat_idx, int) else max(1, min(feat_idx, len(feat_options)))
            feat_choice = feat_options[feat_idx - 1]
            extra_args.append(f"feature.feature_type={feat_choice}")

            # 分子特征参数
            if feat_choice in ["combined", "morgan", "descriptors"]:
                bits_choices = [512, 1024, 2048]
                self.console.print("Morgan bits:")
                for i, b in enumerate(bits_choices, 1):
                    self.console.print(f"  {i}. {b}")
                b_idx = IntPrompt.ask("Bits number", default=2)
                b_idx = 2 if not isinstance(b_idx, int) else max(1, min(b_idx, len(bits_choices)))
                extra_args.append(f"feature.morgan_bits={bits_choices[b_idx-1]}")

                rad_choices = [2, 3]
                self.console.print("Morgan radius:")
                for i, r in enumerate(rad_choices, 1):
                    self.console.print(f"  {i}. {r}")
                r_idx = IntPrompt.ask("Radius number", default=1)
                r_idx = 1 if not isinstance(r_idx, int) else max(1, min(r_idx, len(rad_choices)))
                extra_args.append(f"feature.morgan_radius={rad_choices[r_idx-1]}")

                comb_options = ["mean", "sum", "concat"]
                self.console.print("Combination method:")
                for i, copt in enumerate(comb_options, 1):
                    self.console.print(f"  {i}. {copt}")
                c_idx = IntPrompt.ask("Combination number", default=1)
                c_idx = 1 if not isinstance(c_idx, int) else max(1, min(c_idx, len(comb_options)))
                extra_args.append(f"feature.combination_method={comb_options[c_idx-1]}")

                # SMILES 列（可选）
                set_smiles_idx = IntPrompt.ask("Custom SMILES columns? [1=Yes, 2=No]", default=2)
                if int(set_smiles_idx) == 1:
                    smiles_cols = Prompt.ask("Enter SMILES columns (comma)", default="L1,L2,L3")
                    cols = [c.strip() for c in smiles_cols.split(',') if c.strip()]
                    extra_args.append(f"data.smiles_columns={json.dumps(cols)}")

            # 目标列
            self.console.print("\n[bold]Targets:[/bold]")
            self.console.print("  1. Auto detect")
            self.console.print("  2. Single preset (choose)")
            self.console.print("  3. Custom (comma)")
            tgt_mode = IntPrompt.ask("Target mode", default=1)
            if int(tgt_mode) == 2:
                presets = ["Max_wavelength(nm)", "PLQY", "tau(s*10^-6)"]
                for i, p in enumerate(presets, 1):
                    self.console.print(f"    {i}. {p}")
                p_idx = IntPrompt.ask("Preset number", default=2)
                p_idx = 2 if not isinstance(p_idx, int) else max(1, min(p_idx, len(presets)))
                extra_args.append(f"target={presets[p_idx-1]}")
            elif int(tgt_mode) == 3:
                tgt_input = Prompt.ask("Enter targets (comma)", default="")
                if tgt_input.strip():
                    extra_args.append(f"target={tgt_input.strip()}")

            # 折数与早停
            self.console.print("\n[bold]Cross validation folds:[/bold]")
            folds_options = [3, 5, 10]
            for i, f in enumerate(folds_options, 1):
                self.console.print(f"  {i}. {f}")
            f_idx = IntPrompt.ask("Folds number", default=3)
            f_idx = 3 if not isinstance(f_idx, int) else max(1, min(f_idx, len(folds_options)))
            extra_args.append(f"n_folds={folds_options[f_idx-1]}")

            es_idx = IntPrompt.ask("Enable early stopping? [1=Yes, 2=No]", default=2)
            es_enabled = int(es_idx) == 1
            extra_args.append(f"training.early_stopping={'true' if es_enabled else 'false'}")
            if es_enabled:
                rounds_options = [10, 50, 100]
                for i, rr in enumerate(rounds_options, 1):
                    self.console.print(f"  {i}. rounds={rr}")
                rr_idx = IntPrompt.ask("Early stopping rounds", default=2)
                rr_idx = 2 if not isinstance(rr_idx, int) else max(1, min(rr_idx, len(rounds_options)))
                extra_args.append(f"training.early_stopping_rounds={rounds_options[rr_idx-1]}")

            # 并行/NUMA
            self.console.print("\n[bold]Parallelism:[/bold]")
            par_choices = [1, 2, 4, 8, 16, 32]
            for i, pc in enumerate(par_choices, 1):
                self.console.print(f"  {i}. parallel={pc}")
            p_idx = IntPrompt.ask("Parallel number", default=1)
            p_idx = 1 if not isinstance(p_idx, int) else max(1, min(p_idx, len(par_choices)))
            extra_args.append(f"parallel={par_choices[p_idx-1]}")

            core_choices = [1, 2, 4, 8]
            for i, cc in enumerate(core_choices, 1):
                self.console.print(f"  {i}. cores/task={cc}")
            c_idx = IntPrompt.ask("Cores per task", default=1)
            c_idx = 1 if not isinstance(c_idx, int) else max(1, min(c_idx, len(core_choices)))
            extra_args.append(f"cores={core_choices[c_idx-1]}")

            numa_idx = IntPrompt.ask("Enable NUMA optimization? [1=Yes, 2=No]", default=2)
            extra_args.append(f"numa={'true' if int(numa_idx)==1 else 'false'}")
            bind_idx = IntPrompt.ask("Bind CPU affinity? [1=Yes, 2=No]", default=2)
            extra_args.append(f"bind_cpu={'true' if int(bind_idx)==1 else 'false'}")

            # 测试集（可选）
            test_idx = IntPrompt.ask("Provide test dataset for evaluation? [1=Yes, 2=No]", default=2)
            if int(test_idx) == 1:
                test_file = Prompt.ask("Test data file", default="data/test.csv")
                extra_args.append(f"data.test_data_path={test_file}")

            # 通用保存/报告选项
            self.console.print("\n[bold]Output options:[/bold]")
            save_fold_idx = IntPrompt.ask("Save fold models? [1=Yes, 2=No]", default=2)
            save_fold = int(save_fold_idx) == 1
            extra_args.append(f"training.save_fold_models={'true' if save_fold else 'false'}")

            save_importance_idx = IntPrompt.ask("Save feature importance? [1=Yes, 2=No]", default=1)
            save_importance = int(save_importance_idx) == 1
            extra_args.append(f"training.save_feature_importance={'true' if save_importance else 'false'}")

            gen_report_idx = IntPrompt.ask("Generate analysis report? [1=Yes, 2=No]", default=1)
            gen_report = int(gen_report_idx) == 1
            extra_args.append(f"logging.generate_report={'true' if gen_report else 'false'}")

            # AutoML相关选项（仅当选择automl配置时显示）
            if config.startswith("automl"):
                # 只有在没有选择自定义模型时才询问是否使用所有模型
                if not custom_models_selected:
                    use_all_idx = IntPrompt.ask("AutoML use ALL supported models? [1=Yes, 2=No]", default=1)
                    use_all_models = int(use_all_idx) == 1
                    if use_all_models:
                        extra_args.append("--all")
                
                # AutoML trials / folds / metric (这些选项对自定义模型也适用)
                trials_choices = [20, 50, 100]
                for i, t in enumerate(trials_choices, 1):
                    self.console.print(f"  {i}. trials/model={t}")
                self.console.print(f"  {len(trials_choices)+1}. custom")
                t_idx = IntPrompt.ask("Automl trials per model (choose or custom)", default=2)
                custom_trials = None
                if isinstance(t_idx, int) and t_idx == len(trials_choices)+1:
                    # Custom input
                    try:
                        custom_trials = int(Prompt.ask("Enter custom trials (positive integer)", default="50"))
                    except Exception:
                        custom_trials = 50
                    if custom_trials <= 0:
                        custom_trials = 50
                if custom_trials is not None:
                    extra_args.append(f"optimization.automl_trials_per_model={custom_trials}")
                else:
                    t_idx = 2 if not isinstance(t_idx, int) else max(1, min(t_idx, len(trials_choices)))
                    extra_args.append(f"optimization.automl_trials_per_model={trials_choices[t_idx-1]}")

                optfold_choices = [3, 5, 10]
                for i, of in enumerate(optfold_choices, 1):
                    self.console.print(f"  {i}. optimization folds={of}")
                of_idx = IntPrompt.ask("Optimization folds", default=2)
                of_idx = 2 if not isinstance(of_idx, int) else max(1, min(of_idx, len(optfold_choices)))
                extra_args.append(f"optimization.n_folds={optfold_choices[of_idx-1]}")

                metric_opts = ["rmse", "mae", "r2", "mape"]
                for i, m in enumerate(metric_opts, 1):
                    self.console.print(f"  {i}. metric={m}")
                m_idx = IntPrompt.ask("Optimization metric", default=1)
                m_idx = 1 if not isinstance(m_idx, int) else max(1, min(m_idx, len(metric_opts)))
                metric_ch = metric_opts[m_idx-1]
                extra_args.append(f"optimization.metric={metric_ch}")
                direction = "minimize" if metric_ch in ["rmse","mae","mape"] else "maximize"
                extra_args.append(f"optimization.direction={direction}")

                gen_comp_idx = IntPrompt.ask("Generate comparison table after training? [1=Yes, 2=No]", default=1)
                gen_comp = int(gen_comp_idx) == 1
                extra_args.append(f"comparison.enable={'true' if gen_comp else 'false'}")
                if gen_comp:
                    fmt_options = ["markdown", "html", "latex", "csv"]
                    self.console.print("\n[bold]Comparison formats:[/bold]")
                    for i, fopt in enumerate(fmt_options, 1):
                        self.console.print(f"  {i}. {fopt}")
                    fmt_nums = Prompt.ask("Select formats (numbers, comma), default=1,2,4", default="1,2,4")
                    try:
                        indices = [int(x.strip()) for x in fmt_nums.split(',') if x.strip().isdigit()]
                        indices = [i for i in indices if 1 <= i <= len(fmt_options)]
                        fmts = [fmt_options[i-1] for i in indices] if indices else ["markdown","html","csv"]
                    except Exception:
                        fmts = ["markdown","html","csv"]
                    extra_args.append(f"comparison.formats={json.dumps(fmts)}")

            # 构建命令
            base = [
                "python", "automl.py", "train",
                f"config={config}",
                f"data={data_file}",
                f"project={project}"
            ]
            
            # 显示命令（用于调试）
            cmd_display = " ".join(base + extra_args)
            self.console.print(f"\n[bold]Command:[/bold] {cmd_display}")
            
            if Confirm.ask("Execute training?", default=True):
                # 使用subprocess执行，更好地处理参数
                try:
                    result = subprocess.run(base + extra_args, check=False)
                    if result.returncode == 0:
                        self.current_project = project
                        self.console.print(f"\n✅ Training completed. Project: [bold green]{project}[/bold green]")
                    else:
                        self.console.print(f"\n⚠️ Training exited with code {result.returncode}")
                except Exception as e:
                    self.console.print(f"\n❌ Training failed: {e}")
        else:
            print("Train New Models\n")
            print("This will launch the training pipeline.\n")
            
            # 每个配置支持的模型
            config_models = {
                "xgboost_quick": ["XGBoost"],
                "xgboost_standard": ["XGBoost"],
                "automl_quick": ["XGBoost", "LightGBM", "CatBoost", "Random Forest"],
                "automl": ["XGBoost", "LightGBM", "CatBoost", "Random Forest", "Gradient Boosting", 
                          "Extra Trees", "AdaBoost", "Ridge", "Lasso", "Elastic Net", "SVR", "KNN", "Decision Tree"],
                "paper_comparison": ["XGBoost", "LightGBM", "CatBoost", "Random Forest", "Gradient Boosting", 
                                   "Extra Trees", "AdaBoost", "Ridge", "Lasso", "Elastic Net", "SVR", "KNN", "Decision Tree"],
                "custom": []  # Will be filled by user selection
            }
            
            print("Available configurations:")
            configs = ["xgboost_quick", "xgboost_standard", "automl_quick", "automl", "paper_comparison", "custom"]
            for i, c in enumerate(configs, 1):
                if c == "custom":
                    print(f"  {i}. {c} (Select individual models)")
                else:
                    models = config_models.get(c, [])
                    if len(models) <= 3:
                        models_str = f"({len(models)} models: {', '.join(models)})"
                    else:
                        models_str = f"({len(models)} models: {', '.join(models[:3])}, ...)"
                    print(f"  {i}. {c} {models_str}")
            
            print("\nSupported models by configuration:")
            print("-" * 80)
            for config in configs:
                models = config_models.get(config, [])
                print(f"{config:20} | Count: {len(models):2}")
                # 显示所有模型，分行显示以便阅读
                if len(models) <= 4:
                    print(f"{'':20} | {', '.join(models)}")
                else:
                    # 每行显示4个模型
                    for i in range(0, len(models), 4):
                        chunk = models[i:i+4]
                        if i == 0:
                            print(f"{'':20} | {', '.join(chunk)}")
                        else:
                            print(f"{'':20} | {', '.join(chunk)}")
            print("-" * 80)
            
            print("\nAll Available Models (13 total):")
            print("-" * 80)
            all_models = ["XGBoost", "LightGBM", "CatBoost", "Random Forest", "Gradient Boosting", 
                         "Extra Trees", "AdaBoost", "Ridge", "Lasso", "Elastic Net", "SVR", "KNN", "Decision Tree"]
            
            # 显示为编号列表，每行3个
            for i in range(0, len(all_models), 3):
                row_models = all_models[i:i+3]
                row_str = "  ".join([f"{j+i+1:2}. {model:20}" for j, model in enumerate(row_models)])
                print(row_str)
            print("-" * 80)
            
            config_idx = input("\nSelect configuration (1-6, default=1): ")
            config_idx = int(config_idx) if config_idx else 1
            config = configs[config_idx - 1] if 1 <= config_idx <= len(configs) else configs[0]
            
            # 如果选择了custom，让用户选择具体的模型
            custom_models_list = None
            custom_models_selected = False  # 标记是否选择了自定义模型
            if config == "custom":
                print("\nSelect models to train:")
                print("Enter model numbers separated by commas (e.g., 1,2,3 or 1-5,7,9)\n")
                
                # 显示所有可选模型（注意顺序要与标准模式一致）
                all_model_names = ["adaboost", "catboost", "decision_tree", "elastic_net", 
                                  "extra_trees", "gradient_boosting", "knn", "lasso", 
                                  "lightgbm", "random_forest", "ridge", "svr", "xgboost"]
                all_model_display = ["AdaBoost", "CatBoost", "Decision Tree", "Elastic Net", 
                                    "Extra Trees", "Gradient Boosting", "KNN", "Lasso", 
                                    "LightGBM", "Random Forest", "Ridge", "SVR", "XGBoost"]
                
                # 显示模型列表
                for i, (name, display) in enumerate(zip(all_model_names, all_model_display), 1):
                    print(f"  {i:2}. {display:20} ({name})")
                
                # 获取用户选择
                selection = input("\nSelect models (default=1,2,3,4): ").strip() or "1,2,3,4"
                
                # 解析选择（支持范围如1-5和单个数字）
                selected_indices = []
                for part in selection.split(','):
                    part = part.strip()
                    if '-' in part:
                        try:
                            start, end = map(int, part.split('-'))
                            selected_indices.extend(range(start, end + 1))
                        except:
                            pass
                    else:
                        try:
                            selected_indices.append(int(part))
                        except:
                            pass
                
                # 转换为模型名称
                selected_models = [all_model_names[i-1] for i in selected_indices 
                                 if 1 <= i <= len(all_model_names)]
                
                if not selected_models:
                    selected_models = ["xgboost"]  # 默认至少选择XGBoost
                
                print(f"\nSelected {len(selected_models)} models: {', '.join([all_model_display[all_model_names.index(m)] for m in selected_models])}\n")
                
                # 设置为automl模式但只使用选定的模型
                config = "automl"
                # 保存选定的模型列表，稍后在构建命令时处理
                custom_models_list = selected_models
                custom_models_selected = True  # 标记已选择自定义模型
            
            data_file = input("Data file (default=data/Database_normalized.csv): ")
            data_file = data_file or "data/Database_normalized.csv"
            
            project = input(f"Project name (default=project): ")
            project = project or "project"
            
            # 交互式附加选项（基础终端，编号选择）
            def pick_yn(prompt: str, default_yes: bool = True) -> bool:
                default_num = '1' if default_yes else '2'
                s = input(f"{prompt} [1=Yes, 2=No] (default={default_num}): ").strip()
                if not s:
                    return default_yes
                return s == '1'

            extra_args = []
            if custom_models_list:
                # 使用新的models参数格式（逗号分隔）
                models_str = ','.join(custom_models_list)
                extra_args.append(f"models={models_str}")

            # 特征类型
            feat_options = ["auto", "combined", "morgan", "descriptors", "tabular"]
            print("\nFeature type:")
            for i, fopt in enumerate(feat_options, 1):
                print(f"  {i}. {fopt}")
            s = input("Feature type number (default=2): ").strip() or "2"
            try:
                idx = max(1, min(int(s), len(feat_options)))
            except Exception:
                idx = 2
            feat_choice = feat_options[idx-1]
            extra_args.append(f"feature.feature_type={feat_choice}")

            if feat_choice in ["combined", "morgan", "descriptors"]:
                bits_choices = [512, 1024, 2048]
                print("Morgan bits:")
                for i, b in enumerate(bits_choices, 1):
                    print(f"  {i}. {b}")
                s = input("Bits number (default=2): ").strip() or "2"
                try:
                    bi = max(1, min(int(s), len(bits_choices)))
                except Exception:
                    bi = 2
                extra_args.append(f"feature.morgan_bits={bits_choices[bi-1]}")

                rad_choices = [2, 3]
                print("Morgan radius:")
                for i, r in enumerate(rad_choices, 1):
                    print(f"  {i}. {r}")
                s = input("Radius number (default=1): ").strip() or "1"
                try:
                    ri = max(1, min(int(s), len(rad_choices)))
                except Exception:
                    ri = 1
                extra_args.append(f"feature.morgan_radius={rad_choices[ri-1]}")

                comb_options = ["mean", "sum", "concat"]
                print("Combination method:")
                for i, copt in enumerate(comb_options, 1):
                    print(f"  {i}. {copt}")
                s = input("Combination number (default=1): ").strip() or "1"
                try:
                    ci = max(1, min(int(s), len(comb_options)))
                except Exception:
                    ci = 1
                extra_args.append(f"feature.combination_method={comb_options[ci-1]}")

                # SMILES 列
                set_smiles = pick_yn("Custom SMILES columns?", default_yes=False)
                if set_smiles:
                    smiles_cols = input("Enter SMILES columns (comma, default=L1,L2,L3): ").strip() or "L1,L2,L3"
                    cols = [c.strip() for c in smiles_cols.split(',') if c.strip()]
                    extra_args.append(f"data.smiles_columns={json.dumps(cols)}")

            # 目标列
            print("\nTargets:\n  1. Auto detect\n  2. Single preset (choose)\n  3. Custom (comma)")
            s = input("Target mode (default=1): ").strip() or "1"
            if s == '2':
                presets = ["Max_wavelength(nm)", "PLQY", "tau(s*10^-6)"]
                for i, p in enumerate(presets, 1):
                    print(f"  {i}. {p}")
                s2 = input("Preset number (default=2): ").strip() or "2"
                try:
                    pi = max(1, min(int(s2), len(presets)))
                except Exception:
                    pi = 2
                extra_args.append(f"target={presets[pi-1]}")
            elif s == '3':
                tgt_input = input("Enter targets (comma): ").strip()
                if tgt_input:
                    extra_args.append(f"target={tgt_input}")

            # 折数与早停
            folds_options = [3, 5, 10]
            print("\nCross validation folds:")
            for i, f in enumerate(folds_options, 1):
                print(f"  {i}. {f}")
            s = input("Folds number (default=3): ").strip() or "3"
            try:
                fi = max(1, min(int(s), len(folds_options)))
            except Exception:
                fi = 3
            extra_args.append(f"n_folds={folds_options[fi-1]}")

            es_enabled = pick_yn("Enable early stopping?", default_yes=False)
            extra_args.append(f"training.early_stopping={'true' if es_enabled else 'false'}")
            if es_enabled:
                rounds_options = [10, 50, 100]
                print("Early stopping rounds:")
                for i, rr in enumerate(rounds_options, 1):
                    print(f"  {i}. rounds={rr}")
                s = input("Rounds number (default=2): ").strip() or "2"
                try:
                    ri = max(1, min(int(s), len(rounds_options)))
                except Exception:
                    ri = 2
                extra_args.append(f"training.early_stopping_rounds={rounds_options[ri-1]}")

            # 并行/NUMA
            par_choices = [1, 2, 4, 8, 16, 32]
            print("\nParallel:")
            for i, pc in enumerate(par_choices, 1):
                print(f"  {i}. parallel={pc}")
            s = input("Parallel number (default=1): ").strip() or "1"
            try:
                pi = max(1, min(int(s), len(par_choices)))
            except Exception:
                pi = 1
            extra_args.append(f"parallel={par_choices[pi-1]}")

            core_choices = [1, 2, 4, 8]
            for i, cc in enumerate(core_choices, 1):
                print(f"  {i}. cores/task={cc}")
            s = input("Cores per task (default=1): ").strip() or "1"
            try:
                ci = max(1, min(int(s), len(core_choices)))
            except Exception:
                ci = 1
            extra_args.append(f"cores={core_choices[ci-1]}")

            numa = pick_yn("Enable NUMA optimization?", default_yes=False)
            extra_args.append(f"numa={'true' if numa else 'false'}")
            bind = pick_yn("Bind CPU affinity?", default_yes=False)
            extra_args.append(f"bind_cpu={'true' if bind else 'false'}")

            # 测试集
            provide_test = pick_yn("Provide test dataset for evaluation?", default_yes=False)
            if provide_test:
                test_file = input("Test data file (default=data/test.csv): ").strip() or "data/test.csv"
                extra_args.append(f"data.test_data_path={test_file}")
            save_fold = pick_yn("Save fold models?", default_yes=False)
            extra_args.append(f"training.save_fold_models={'true' if save_fold else 'false'}")
            save_importance = pick_yn("Save feature importance?", default_yes=True)
            extra_args.append(f"training.save_feature_importance={'true' if save_importance else 'false'}")
            gen_report = pick_yn("Generate analysis report?", default_yes=True)
            extra_args.append(f"logging.generate_report={'true' if gen_report else 'false'}")

            if config.startswith("automl"):
                # 只有在没有选择自定义模型时才询问是否使用所有模型
                if not custom_models_selected:
                    use_all_models = pick_yn("AutoML use ALL supported models?", default_yes=True)
                    if use_all_models:
                        extra_args.append("--all")
                
                # AutoML trials / folds / metric (这些选项对自定义模型也适用)
                trials_choices = [20, 50, 100]
                print("Automl trials per model:")
                for i, t in enumerate(trials_choices, 1):
                    print(f"  {i}. {t}")
                print(f"  {len(trials_choices)+1}. custom")
                s = input("Trials number (choose index or custom, default=2): ").strip() or "2"
                custom_trials = None
                if s.isdigit() and int(s) == len(trials_choices)+1:
                    cs = input("Enter custom trials (positive integer, default=50): ").strip() or "50"
                    try:
                        custom_trials = int(cs)
                    except Exception:
                        custom_trials = 50
                    if custom_trials <= 0:
                        custom_trials = 50
                if custom_trials is not None:
                    extra_args.append(f"optimization.automl_trials_per_model={custom_trials}")
                else:
                    try:
                        ti = max(1, min(int(s), len(trials_choices)))
                    except Exception:
                        ti = 2
                    extra_args.append(f"optimization.automl_trials_per_model={trials_choices[ti-1]}")

                optfold_choices = [3, 5, 10]
                print("Optimization folds:")
                for i, of in enumerate(optfold_choices, 1):
                    print(f"  {i}. {of}")
                s = input("Opt folds number (default=2): ").strip() or "2"
                try:
                    ofi = max(1, min(int(s), len(optfold_choices)))
                except Exception:
                    ofi = 2
                extra_args.append(f"optimization.n_folds={optfold_choices[ofi-1]}")

                metric_opts = ["rmse", "mae", "r2", "mape"]
                print("Optimization metric:")
                for i, m in enumerate(metric_opts, 1):
                    print(f"  {i}. {m}")
                s = input("Metric number (default=1): ").strip() or "1"
                try:
                    mi = max(1, min(int(s), len(metric_opts)))
                except Exception:
                    mi = 1
                metric_ch = metric_opts[mi-1]
                extra_args.append(f"optimization.metric={metric_ch}")
                direction = "minimize" if metric_ch in ["rmse","mae","mape"] else "maximize"
                extra_args.append(f"optimization.direction={direction}")

                gen_comp = pick_yn("Generate comparison table after training?", default_yes=True)
                extra_args.append(f"comparison.enable={'true' if gen_comp else 'false'}")
                if gen_comp:
                    fmt_options = ["markdown", "html", "latex", "csv"]
                    print("\nComparison formats:")
                    for i, fopt in enumerate(fmt_options, 1):
                        print(f"  {i}. {fopt}")
                    fmt_nums = input("Select formats (numbers, comma, default=1,2,4): ").strip() or "1,2,4"
                    try:
                        indices = [int(x.strip()) for x in fmt_nums.split(',') if x.strip().isdigit()]
                        indices = [i for i in indices if 1 <= i <= len(fmt_options)]
                        fmts = [fmt_options[i-1] for i in indices] if indices else ["markdown","html","csv"]
                    except Exception:
                        fmts = ["markdown","html","csv"]
                    extra_args.append(f"comparison.formats={json.dumps(fmts)}")

            base = [
                "python", "automl.py", "train",
                f"config={config}",
                f"data={data_file}",
                f"project={project}"
            ]
            # 显示命令（用于调试）
            cmd_display = " ".join(base + extra_args)
            print(f"\nCommand: {cmd_display}")
            
            proceed = input("Execute training? (y/n, default=y): ")
            if proceed.lower() != 'n':
                # 使用subprocess执行，更好地处理参数
                try:
                    result = subprocess.run(base + extra_args, check=False)
                    if result.returncode == 0:
                        self.current_project = project
                        print(f"\n✅ Training completed. Project: {project}")
                    else:
                        print(f"\n⚠️ Training exited with code {result.returncode}")
                except Exception as e:
                    print(f"\n❌ Training failed: {e}")
    
    def clean_project(self):
        """清理项目"""
        if not self.current_project:
            if RICH_AVAILABLE:
                self.console.print("[yellow]Please select a project first.[/yellow]")
            else:
                print("Please select a project first.")
            return
        
        self.print_header()
        
        if RICH_AVAILABLE:
            self.console.print(f"[bold]Clean Project: {self.current_project}[/bold]\n")
            
            keep_models = Confirm.ask("Keep model files?", default=True)
            keep_results = Confirm.ask("Keep result files?", default=True)
            
            if Confirm.ask(f"\n[red]This will delete intermediate files. Continue?[/red]", default=False):
                try:
                    self.manager.clean_project(self.current_project, keep_models, keep_results)
                    self.console.print("✅ [green]Project cleaned successfully.[/green]")
                except Exception as e:
                    self.console.print(f"[red]Clean failed: {e}[/red]")
        else:
            print(f"Clean Project: {self.current_project}\n")
            
            keep_models = input("Keep model files? (y/n, default=y): ")
            keep_models = keep_models.lower() != 'n'
            
            keep_results = input("Keep result files? (y/n, default=y): ")
            keep_results = keep_results.lower() != 'n'
            
            confirm = input("\nThis will delete intermediate files. Continue? (y/n): ")
            if confirm.lower() == 'y':
                try:
                    self.manager.clean_project(self.current_project, keep_models, keep_results)
                    print("✅ Project cleaned successfully.")
                except Exception as e:
                    print(f"Clean failed: {e}")
    
    def run(self):
        """运行交互式界面"""
        while True:
            choice = self.main_menu()
            
            if choice == "0":
                if RICH_AVAILABLE:
                    self.console.print("\n[bold cyan]Goodbye![/bold cyan] 👋")
                else:
                    print("\nGoodbye! 👋")
                break
            elif choice == "1":
                self.list_projects()
            elif choice == "2":
                self.select_project()
            elif choice == "3":
                self.show_project_info()
            elif choice == "4":
                self.batch_prediction()
            elif choice == "5":
                self.train_models()
            elif choice == "6":
                self.view_comparison_table()
            elif choice == "7":
                self.export_project()
            elif choice == "8":
                self.generate_comparison_table()
            elif choice == "9":
                self.generate_report()
            elif choice == "10":
                self.clean_project()
            
            if choice != "0":
                if RICH_AVAILABLE:
                    self.console.input("\n[dim]Press Enter to continue...[/dim]")
                else:
                    input("\nPress Enter to continue...")


def main():
    """主函数"""
    cli = InteractiveCLI()
    try:
        cli.run()
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            cli.console.print("\n\n[yellow]Interrupted by user.[/yellow]")
        else:
            print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        if RICH_AVAILABLE:
            cli.console.print(f"\n[red]Error: {e}[/red]")
        else:
            print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
