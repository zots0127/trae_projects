#!/usr/bin/env python3
"""
Interactive CLI management interface
Provides a user-friendly project management and batch prediction interface
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

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Try importing rich; fall back to basic interface if unavailable
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
    print("Warning: Rich library not installed. Using basic interface.")
    print("Install with: pip install rich")

from utils.project_manager import ProjectManager
from utils.project_predictor import ProjectPredictor


class InteractiveCLI:
    """Interactive CLI management interface"""
    
    def __init__(self):
        """Initialize CLI"""
        self.console = Console() if RICH_AVAILABLE else None
        self.manager = ProjectManager()
        self.current_project = None
        self.current_predictor = None
        
    def clear_screen(self):
        """Clear screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print header"""
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
        """Main menu"""
        self.print_header()
        
        if self.current_project:
            if RICH_AVAILABLE:
                self.console.print(f"\nCurrent Project: [bold green]{self.current_project}[/bold green]\n")
            else:
                print(f"\nCurrent Project: {self.current_project}\n")
        
        menu_items = [
            "1. List Projects",
            "2. Select Project",
            "3. Project Information",
            "4. Batch Prediction",
            "5. Train New Models",
            "6. View Comparison Table",
            "7. Export Project",
            "8. Generate Comparison Table",
            "9. Generate Report",
            "10. Clean Project",
            "0. Exit"
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
        """List all projects"""
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
        """Select project"""
        self.print_header()
        projects = self.manager.list_projects()
        
        if not projects:
            if RICH_AVAILABLE:
                self.console.print("[yellow]No projects found.[/yellow]")
            else:
                print("No projects found.")
            return
        
        # Show project list
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
                self.console.print(f"\nSelected project: [bold green]{self.current_project}[/bold green]")
            else:
                print(f"\nSelected project: {self.current_project}")
        else:
            if RICH_AVAILABLE:
                self.console.print("[red]Invalid selection.[/red]")
            else:
                print("Invalid selection.")
    
    def show_project_info(self):
        """Show project information"""
        if not self.current_project:
            if RICH_AVAILABLE:
                self.console.print("[yellow]Please select a project first.[/yellow]")
            else:
                print("Please select a project first.")
            return
        
        self.print_header()
        
        try:
            # Get project info
            info = self.manager.get_project_info(self.current_project)
            
            # Load predictor to get model details
            if not self.current_predictor:
                self.current_predictor = ProjectPredictor(self.current_project, verbose=False)
            
            if RICH_AVAILABLE:
                # Basic info
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
                
                # Model list
                if self.current_predictor.models:
                    table = Table(title="Trained Models", show_header=True, header_style="bold cyan")
                    table.add_column("Model", style="cyan")
                    table.add_column("Target", style="green")
                    table.add_column("R^2 (mean+/-std)", justify="right", style="yellow")
                    table.add_column("RMSE (mean+/-std)", justify="right", style="yellow")
                    table.add_column("MAE (mean+/-std)", justify="right", style="yellow")
                    
                    for key, info in self.current_predictor.models.items():
                        perf = info.get('performance', {})
                        
                        # Format R^2 with std
                        r2_str = 'N/A'
                        if isinstance(perf.get('r2'), float):
                            r2_mean = perf.get('r2')
                            r2_std = perf.get('r2_std', 0)
                            if r2_std > 0:
                                r2_str = f"{r2_mean:.4f}+/-{r2_std:.4f}"
                            else:
                                r2_str = f"{r2_mean:.4f}"
                        
                        # Format RMSE with std
                        rmse_str = 'N/A'
                        if isinstance(perf.get('rmse'), float):
                            rmse_mean = perf.get('rmse')
                            rmse_std = perf.get('rmse_std', 0)
                            if rmse_std > 0:
                                rmse_str = f"{rmse_mean:.2f}+/-{rmse_std:.2f}"
                            else:
                                rmse_str = f"{rmse_mean:.2f}"
                        
                        # Format MAE with std
                        mae_str = 'N/A'
                        if isinstance(perf.get('mae'), float):
                            mae_mean = perf.get('mae')
                            mae_std = perf.get('mae_std', 0)
                            if mae_std > 0:
                                mae_str = f"{mae_mean:.2f}+/-{mae_std:.2f}"
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
                
                # Best models
                if info.get('best_models'):
                    best_panel = Panel(
                        "\n".join([
                            f"[bold]{target}:[/bold] {best['model']} (R^2={best['r2']:.4f})"
                            for target, best in info['best_models'].items()
                        ]),
                        title="Best Models",
                        border_style="yellow"
                    )
                    self.console.print(best_panel)
            else:
                # Basic text output
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
                        print(f"    R^2={perf.get('r2', 'N/A'):.4f}, RMSE={perf.get('rmse', 'N/A'):.4f}")
                
                if info.get('best_models'):
                    print("\nBest Models:")
                    for target, best in info['best_models'].items():
                        print(f"  {target}: {best['model']} (R^2={best['r2']:.4f})")
                        
        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"[red]Error: {e}[/red]")
            else:
                print(f"Error: {e}")
    
    def batch_prediction(self):
        """Batch prediction interactive flow"""
        if not self.current_project:
            if RICH_AVAILABLE:
                self.console.print("[yellow]Please select a project first.[/yellow]")
            else:
                print("Please select a project first.")
            return
        
        self.print_header()
        
        # Get data file
        if RICH_AVAILABLE:
            self.console.print("[bold]Batch Prediction Setup[/bold]\n")
            data_file = Prompt.ask("Enter data file path", default="data/Database_normalized.csv")
        else:
            print("\nBatch Prediction Setup\n")
            data_file = input("Enter data file path (default=data/Database_normalized.csv): ")
            data_file = data_file or "data/Database_normalized.csv"
        
        # Check file existence
        if not Path(data_file).exists():
            if RICH_AVAILABLE:
                self.console.print(f"[red]File not found: {data_file}[/red]")
            else:
                print(f"File not found: {data_file}")
            return
        
        # Choose prediction mode
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
        
        # If ensemble, ask for method
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
        
        # Output file
        default_output = f"{self.current_project}/predictions_{mode}.csv"
        
        if RICH_AVAILABLE:
            output_file = Prompt.ask("Output file", default=default_output)
        else:
            output_file = input(f"Output file (default={default_output}): ")
            output_file = output_file or default_output
        
        # Confirm execution
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
        
        # Run prediction
        try:
            if RICH_AVAILABLE:
                with self.console.status("[bold green]Running prediction...", spinner="dots"):
                    self._run_prediction(data_file, mode, output_file, method)
            else:
                print("\nRunning prediction...")
                self._run_prediction(data_file, mode, output_file, method)
            
            if RICH_AVAILABLE:
                self.console.print(f"\n[green]Prediction completed.[/green]")
                self.console.print(f"   Output saved to: {output_file}")
            else:
                print(f"\nPrediction completed.")
                print(f"   Output saved to: {output_file}")
                
        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"[red]Prediction failed: {e}[/red]")
            else:
                print(f"Prediction failed: {e}")
    
    def _run_prediction(self, data_file: str, mode: str, output_file: str, method: str = "mean"):
        """Run prediction"""
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
        """View comparison table"""
        if not self.current_project:
            if RICH_AVAILABLE:
                self.console.print("[yellow]Please select a project first.[/yellow]")
            else:
                print("Please select a project first.")
            return
        
        self.print_header()
        
        # Find comparison table files
        project_path = Path(self.current_project)
        table_files = list(project_path.glob("comparison_table_*.csv"))
        
        if not table_files:
            # If no table exists, prompt to generate immediately
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

            # Reload
            table_files = list(project_path.glob("comparison_table_*.csv"))
            if not table_files:
                if RICH_AVAILABLE:
                    self.console.print("[yellow]Still no tables found after generation.[/yellow]")
                else:
                    print("Still no tables found after generation.")
                return
        
        # Read the latest table
        latest_table = sorted(table_files)[-1]
        df = pd.read_csv(latest_table)
        
        if RICH_AVAILABLE:
            self.console.print(f"[bold]Comparison Table:[/bold] {latest_table.name}\n")
            
            # Convert to rich table
            table = Table(show_header=True, header_style="bold magenta", title=f"Model Comparison Table")
            
            # Smart add columns - detect mean/std
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
            
            # Add headers
            for col in df.columns:
                # Check if paired mean/std columns
                is_metric_col = False
                for base_name, group in col_groups.items():
                    if col == group.get('mean'):
                        # Mean column; check corresponding std
                        if 'std' in group:
                            table.add_column(f"{base_name} (mean+/-std)", justify="right", style="yellow")
                        else:
                            table.add_column(col, justify="right", style="yellow")
                        is_metric_col = True
                        break
                    elif col == group.get('std'):
                        # Std column merged with mean; skip
                        is_metric_col = True
                        break
                
                if not is_metric_col:
                    # Regular column
                    if 'R2' in col or 'RMSE' in col or 'MAE' in col:
                        table.add_column(col, justify="right", style="yellow")
                    else:
                        table.add_column(col, style="cyan")
            
            # Add rows
            for _, row in df.iterrows():
                row_data = []
                processed_cols = set()
                
                for col in df.columns:
                    if col in processed_cols:
                        continue
                    
                    # Check whether to merge mean/std
                    merged = False
                    for base_name, group in col_groups.items():
                        if col == group.get('mean') and 'std' in group:
                            mean_val = row[col]
                            std_val = row[group['std']]
                            if isinstance(mean_val, float) and isinstance(std_val, float):
                                row_data.append(f"{mean_val:.4f}+/-{std_val:.4f}")
                            else:
                                row_data.append(str(mean_val))
                            processed_cols.add(col)
                            processed_cols.add(group['std'])
                            merged = True
                            break
                    
                    if not merged and col not in processed_cols:
                        # Standalone std column (should be processed)
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
        """Generate model comparison table (integrated with Manager)"""
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
                self.console.print("\n[green]Comparison table generated.[/green]")
                for k, v in exported.items():
                    self.console.print(f"   - {k}: {v}")
                # Render once in console after generation
                try:
                    self.console.print("\n[bold]Preview (rendered in console):[/bold]\n")
                    self.view_comparison_table()
                except Exception:
                    pass
            else:
                print("\nComparison table generated.")
                for k, v in exported.items():
                    print(f"   - {k}: {v}")
                # Render once in console after generation
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
        """Export project"""
        if not self.current_project:
            if RICH_AVAILABLE:
                self.console.print("[yellow]Please select a project first.[/yellow]")
            else:
                print("Please select a project first.")
            return
        
        self.print_header()
        
        # Select format
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
                self.console.print(f"[green]Project exported to: {output_file}[/green]")
            else:
                print(f"Project exported to: {output_file}")
                
        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"[red]Export failed: {e}[/red]")
            else:
                print(f"Export failed: {e}")
    
    def generate_report(self):
        """Generate report"""
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
                self.console.print(f"[green]Report generated: {output_file}[/green]")
            else:
                print(f"Report generated: {output_file}")
                
        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"[red]Report generation failed: {e}[/red]")
            else:
                print(f"Report generation failed: {e}")
    
    def train_models(self):
        """Train new models"""
        self.print_header()
        
        if RICH_AVAILABLE:
            self.console.print("[bold]Train New Models[/bold]\n")
            self.console.print("This will launch the training pipeline.\n")
            
            # Configuration selection by number - add model info display and custom option
            configs = ["xgboost_quick", "xgboost_standard", "automl_quick", "automl", "paper_comparison", "custom"]
            
            # Models supported by each configuration
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
            
            # Display detailed model list
            self.console.print("\n[bold]Supported models by configuration:[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Config", style="cyan")
            table.add_column("Models", style="yellow", overflow="fold")
            table.add_column("Count", justify="center", style="green")
            
            for config in configs:
                models = config_models.get(config, [])
                # Show all models without truncation
                models_display = ", ".join(models)
                table.add_row(config, models_display, str(len(models)))
            
            self.console.print(table)
            
            all_models = [
                ("XGBoost", "Tree-based ensemble"),
                ("LightGBM", "Tree-based ensemble"),
                ("CatBoost", "Tree-based ensemble"),
                ("Random Forest", "Tree-based ensemble"),
                ("Gradient Boosting", "Tree-based ensemble"),
                ("Extra Trees", "Tree-based ensemble"),
                ("AdaBoost", "Tree-based ensemble"),
                ("Decision Tree", "Single tree"),
                ("Ridge", "Linear model"),
                ("Lasso", "Linear model"),
                ("Elastic Net", "Linear model"),
                ("SVR", "Support Vector"),
                ("KNN", "Instance-based")
            ]
            self.console.print(f"\n[bold]All Available Models ({len(all_models)} total):[/bold]")
            
            # Create a table to display all models
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
            
            # If custom is selected, let user choose specific models
            custom_models_list = None
            custom_models_selected = False  # Mark whether custom models are selected
            if config == "custom":
                self.console.print("\n[bold]Select models to train:[/bold]")
                self.console.print("[dim]Enter model numbers separated by commas (e.g., 1,2,3 or 1-5,7,9)[/dim]\n")
                
                # Show all selectable models (order consistent with standard mode)
                all_model_names = ["adaboost", "catboost", "decision_tree", "elastic_net", 
                                  "extra_trees", "gradient_boosting", "knn", "lasso", 
                                  "lightgbm", "random_forest", "ridge", "svr", "xgboost"]
                all_model_display = ["AdaBoost", "CatBoost", "Decision Tree", "Elastic Net", 
                                    "Extra Trees", "Gradient Boosting", "KNN", "Lasso", 
                                    "LightGBM", "Random Forest", "Ridge", "SVR", "XGBoost"]
                
                # Create model selection table
                for i, (name, display) in enumerate(zip(all_model_names, all_model_display), 1):
                    self.console.print(f"  {i:2}. [yellow]{display:20}[/yellow] [dim]({name})[/dim]")
                
                # Get user selection
                selection = Prompt.ask("\nSelect models", default="1,2,3,4")
                
                # Parse selection (supports ranges like 1-5 and single numbers)
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
                
                # Convert to model names
                selected_models = [all_model_names[i-1] for i in selected_indices 
                                 if 1 <= i <= len(all_model_names)]
                
                if not selected_models:
                    selected_models = ["xgboost"]  # Default selects at least XGBoost
                
                self.console.print(f"\n[green]Selected {len(selected_models)} models:[/green] {', '.join([all_model_display[all_model_names.index(m)] for m in selected_models])}\n")
                
                # Set to automl mode but only use selected models
                config = "automl"
                # Save selected model list for later command building
                custom_models_list = selected_models
                custom_models_selected = True  # Mark custom models selected
            
            # Get data file
            data_file = Prompt.ask("Data file", default="data/Database_normalized.csv")
            
            # Get project name
            project = Prompt.ask("Project name", default="project")
            
            # Interactive extra options (numeric selection)
            extra_args = []
            if custom_models_list:
                # Use new models parameter format (comma-separated)
                models_str = ','.join(custom_models_list)
                extra_args.append(f"models={models_str}")

            # Feature type
            feat_options = ["auto", "combined", "morgan", "descriptors", "tabular"]
            self.console.print("\n[bold]Feature type:[/bold]")
            for i, fopt in enumerate(feat_options, 1):
                self.console.print(f"  {i}. {fopt}")
            feat_idx = IntPrompt.ask("Feature type number", default=2)
            feat_idx = 2 if not isinstance(feat_idx, int) else max(1, min(feat_idx, len(feat_options)))
            feat_choice = feat_options[feat_idx - 1]
            extra_args.append(f"feature.feature_type={feat_choice}")

            # Molecular feature parameters
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

                # SMILES columns (optional)
                set_smiles_idx = IntPrompt.ask("Custom SMILES columns? [1=Yes, 2=No]", default=2)
                if int(set_smiles_idx) == 1:
                    smiles_cols = Prompt.ask("Enter SMILES columns (comma)", default="L1,L2,L3")
                    cols = [c.strip() for c in smiles_cols.split(',') if c.strip()]
                    extra_args.append(f"data.smiles_columns={json.dumps(cols)}")

            # Target columns
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

            # Folds and early stopping
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

            # Parallel/NUMA
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

            # Test dataset (optional)
            test_idx = IntPrompt.ask("Provide test dataset for evaluation? [1=Yes, 2=No]", default=2)
            if int(test_idx) == 1:
                test_file = Prompt.ask("Test data file", default="data/test.csv")
                extra_args.append(f"data.test_data_path={test_file}")

            # Common save/report options
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

            # AutoML options (shown only for automl configuration)
            if config.startswith("automl"):
                # Ask to use all models only when custom models not selected
                if not custom_models_selected:
                    use_all_idx = IntPrompt.ask("AutoML use ALL supported models? [1=Yes, 2=No]", default=1)
                    use_all_models = int(use_all_idx) == 1
                    if use_all_models:
                        extra_args.append("--all")
                
                # AutoML trials / folds / metric (also applicable to custom models)
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

            # Build command (no external config; translate selection to CLI params)
            base = [
                "python", "automl.py", "train",
                f"data={data_file}",
                f"project={project}"
            ]
            # Map configuration to CLI model selection
            if config == "xgboost_quick" or config == "xgboost_standard":
                base.append("model=xgboost")
            elif config == "automl_quick":
                base.append("models=xgboost,lightgbm,catboost,random_forest")
            elif config == "automl" or config == "paper_comparison":
                base.append("models=adaboost,catboost,decision_tree,elastic_net,gradient_boosting,knn,lasso,lightgbm,mlp,random_forest,ridge,svr,xgboost")
            elif config == "custom" and custom_models_selected and custom_models_list:
                base.append("models=" + ",".join(custom_models_list))
            
            # Display command (for debugging)
            cmd_display = " ".join(base + extra_args)
            self.console.print(f"\n[bold]Command:[/bold] {cmd_display}")
            
            if Confirm.ask("Execute training?", default=True):
                # Use subprocess for better argument handling
                try:
                    result = subprocess.run(base + extra_args, check=False)
                    if result.returncode == 0:
                        self.current_project = project
                        self.console.print(f"\nTraining completed. Project: [bold green]{project}[/bold green]")
                    else:
                        self.console.print(f"\nTraining exited with code {result.returncode}")
                except Exception as e:
                    self.console.print(f"\nTraining failed: {e}")
        else:
            print("Train New Models\n")
            print("This will launch the training pipeline.\n")
            
            # Supported models per configuration
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
                # Show all models across lines for readability
                if len(models) <= 4:
                    print(f"{'':20} | {', '.join(models)}")
                else:
                    # Display 4 models per line
                    for i in range(0, len(models), 4):
                        chunk = models[i:i+4]
                        if i == 0:
                            print(f"{'':20} | {', '.join(chunk)}")
                        else:
                            print(f"{'':20} | {', '.join(chunk)}")
            print("-" * 80)
            
            all_models = ["XGBoost", "LightGBM", "CatBoost", "Random Forest", "Gradient Boosting", 
                         "Extra Trees", "AdaBoost", "Ridge", "Lasso", "Elastic Net", "SVR", "KNN", "Decision Tree"]
            print(f"\nAll Available Models ({len(all_models)} total):")
            print("-" * 80)
            
            # Show as numbered list, 3 per line
            for i in range(0, len(all_models), 3):
                row_models = all_models[i:i+3]
                row_str = "  ".join([f"{j+i+1:2}. {model:20}" for j, model in enumerate(row_models)])
                print(row_str)
            print("-" * 80)
            
            config_idx = input("\nSelect configuration (1-6, default=1): ")
            config_idx = int(config_idx) if config_idx else 1
            config = configs[config_idx - 1] if 1 <= config_idx <= len(configs) else configs[0]
            
            # If 'custom' selected, let user choose specific models
            custom_models_list = None
            custom_models_selected = False  # Flag whether custom models were selected
            if config == "custom":
                print("\nSelect models to train:")
                print("Enter model numbers separated by commas (e.g., 1,2,3 or 1-5,7,9)\n")
                
                # Display all selectable models (keep order consistent with standard mode)
                all_model_names = ["adaboost", "catboost", "decision_tree", "elastic_net", 
                                  "extra_trees", "gradient_boosting", "knn", "lasso", 
                                  "lightgbm", "random_forest", "ridge", "svr", "xgboost"]
                all_model_display = ["AdaBoost", "CatBoost", "Decision Tree", "Elastic Net", 
                                    "Extra Trees", "Gradient Boosting", "KNN", "Lasso", 
                                    "LightGBM", "Random Forest", "Ridge", "SVR", "XGBoost"]
                
                # Display model list
                for i, (name, display) in enumerate(zip(all_model_names, all_model_display), 1):
                    print(f"  {i:2}. {display:20} ({name})")
                
                # Get user selection
                selection = input("\nSelect models (default=1,2,3,4): ").strip() or "1,2,3,4"
                
                # Parse selection (supports ranges like 1-5 and single numbers)
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
                
                # Convert to model names
                selected_models = [all_model_names[i-1] for i in selected_indices 
                                 if 1 <= i <= len(all_model_names)]
                
                if not selected_models:
                    selected_models = ["xgboost"]  # Default selects at least XGBoost
                
                print(f"\nSelected {len(selected_models)} models: {', '.join([all_model_display[all_model_names.index(m)] for m in selected_models])}\n")
                
                # Set to automl mode but only use selected models
                config = "automl"
                # Save selected model list for later command building
                custom_models_list = selected_models
                custom_models_selected = True  # Mark custom models selected
            
            data_file = input("Data file (default=data/Database_normalized.csv): ")
            data_file = data_file or "data/Database_normalized.csv"
            
            project = input(f"Project name (default=project): ")
            project = project or "project"
            
            # Interactive extra options (basic terminal, numeric selection)
            def pick_yn(prompt: str, default_yes: bool = True) -> bool:
                default_num = '1' if default_yes else '2'
                s = input(f"{prompt} [1=Yes, 2=No] (default={default_num}): ").strip()
                if not s:
                    return default_yes
                return s == '1'

            extra_args = []
            if custom_models_list:
                # Use new models parameter format (comma-separated)
                models_str = ','.join(custom_models_list)
                extra_args.append(f"models={models_str}")

            # Feature type
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

                # SMILES columns
                set_smiles = pick_yn("Custom SMILES columns?", default_yes=False)
                if set_smiles:
                    smiles_cols = input("Enter SMILES columns (comma, default=L1,L2,L3): ").strip() or "L1,L2,L3"
                    cols = [c.strip() for c in smiles_cols.split(',') if c.strip()]
                    extra_args.append(f"data.smiles_columns={json.dumps(cols)}")

            # Target columns
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

            # Folds and early stopping
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

            # Parallel/NUMA
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

            # Test dataset
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
                # Ask to use all models only when custom models not selected
                if not custom_models_selected:
                    use_all_models = pick_yn("AutoML use ALL supported models?", default_yes=True)
                    if use_all_models:
                        extra_args.append("--all")
                
                # AutoML trials / folds / metric (also applicable to custom models)
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
            # Display command (for debugging)
            cmd_display = " ".join(base + extra_args)
            print(f"\nCommand: {cmd_display}")
            
            proceed = input("Execute training? (y/n, default=y): ")
            if proceed.lower() != 'n':
                # Use subprocess for better argument handling
                try:
                    result = subprocess.run(base + extra_args, check=False)
                    if result.returncode == 0:
                        self.current_project = project
                        print(f"\nTraining completed. Project: {project}")
                    else:
                        print(f"\nTraining exited with code {result.returncode}")
                except Exception as e:
                    print(f"\nTraining failed: {e}")
    
    def clean_project(self):
        """Clean project"""
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
                    self.console.print("[green]Project cleaned successfully.[/green]")
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
                    print("Project cleaned successfully.")
                except Exception as e:
                    print(f"Clean failed: {e}")
    
    def run(self):
        """Run interactive interface"""
        while True:
            choice = self.main_menu()
            
            if choice == "0":
                if RICH_AVAILABLE:
                    self.console.print("\n[bold cyan]Goodbye![/bold cyan]")
                else:
                    print("\nGoodbye!")
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
    """Main function"""
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
