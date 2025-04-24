import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from tqdm import tqdm
import gc
from collections import defaultdict


class SimulationAnalyzer:
    def __init__(self, results_dir="SimulationResults"):
        self.results_dir = results_dir
        self.metrics = {
            "total_throughput": [],
            "avg_user_throughput": [],
            "gbr_throughput": [],
            "non_gbr_throughput": [],
            "avg_delay": [],
            "jitter": [],
            "packet_loss": [],
            "spectral_efficiency": [],
            "fairness_index": [],
            "scheduling_efficiency": [],
            "qos_compliance": [],
        }
        self.simulation_params = []
        self.user_qos_stats = defaultdict(list)
        self.user_actual_qos_values = defaultdict(list)  # New dictionary to store actual QoS values

    def parse_folder_name(self, folder_name):
        """Извлекает параметры симуляции из названия папки"""
        params = {}
        scenario_part, *params_parts = folder_name.split("_", 1)
        params["scenario"] = scenario_part

        if len(params_parts) == 0:
            return params

        param_mapping = {
            "BW": "bandwidth",
            "UEs": "ue_count",
            "TM": "transmission_mode",
            "CA": "carrier_aggregation",
            "Sched": "scheduler",
            "Mob": "mobility",
            "Dist": "distance",
            "GBR": "gbr_traffic",
            "NonGBR": "non_gbr_traffic",
        }

        parts = params_parts[0].split("_")

        for part in parts:
            for key in param_mapping:
                if part.startswith(key):
                    param_name = param_mapping[key]
                    param_value = part[len(key) :]

                    if key in ["BW", "UEs", "TM", "Dist"]:
                        try:
                            param_value = int(param_value)
                        except ValueError:
                            pass
                    elif key == "CA":
                        param_value = param_value.lower() == "on"

                    params[param_name] = param_value

        return params

    def read_simulation_file(self, filepath):
        """Читает файл с результатами симуляции"""
        try:
            try:
                df = pd.read_csv(filepath, sep="\t")
            except:
                df = pd.read_csv(filepath)

            df = df.reset_index(drop=True)
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None

    def calculate_qos_compliance(self, dl_pdcp, ue_count):
        """Вычисляет соответствие QoS требованиям с обновленными параметрами"""
        if dl_pdcp is None or "IMSI" not in dl_pdcp.columns:
            return 0.0

        qos_requirements = {
            "gbr": {  # GBR traffic (even IMSI) - video
                "throughput": 5.0,  # Mbps (required for video)
                "delay": 150.0,  # ms Packet Delay Budget
                "loss": 0.1,  # 10^-3 Packet Loss Rate (0.1%)
            },
            "non_gbr": {  # Non-GBR traffic (odd IMSI) - voice
                "throughput": 1.0,  # Mbps (required for voice)
                "delay": 100.0,  # ms Packet Delay Budget
                "loss": 0.1,  # 10^-3 Packet Loss Rate (0.1%)
            },
        }

        compliance_stats = {
            "gbr": {"throughput": 0, "delay": 0, "loss": 0, "total": 0},
            "non_gbr": {"throughput": 0, "delay": 0, "loss": 0, "total": 0},
        }

        actual_values = {
            "gbr": {"throughput": [], "delay": [], "loss": []},
            "non_gbr": {"throughput": [], "delay": [], "loss": []},
        }

        try:
            grouped = dl_pdcp.groupby("IMSI")

            for imsi, group in grouped:
                if imsi % 2 == 1:
                    traffic_type = "gbr"
                else:
                    traffic_type = "non_gbr"

                req = qos_requirements[traffic_type]
                compliance_stats[traffic_type]["total"] += 1

                time_diff = group["end"].max() - group["% start"].min()
                if time_diff <= 0:
                    time_diff = 1.0

                user_throughput = group["TxBytes"].sum() * 8 / time_diff / 1e6
                user_delay = group["delay"].mean() if "delay" in group.columns else 0

                tx_packets = group["TxBytes"].sum() if "TxBytes" in group.columns else 1
                rx_packets = group["RxBytes"].sum() if "RxBytes" in group.columns else 0
                user_loss = (tx_packets - rx_packets) / tx_packets * 100 if tx_packets > 0 else 0

                # Store actual values
                actual_values[traffic_type]["throughput"].append(user_throughput)
                actual_values[traffic_type]["delay"].append(user_delay)
                actual_values[traffic_type]["loss"].append(user_loss)

                # Check QoS compliance
                if user_throughput >= req["throughput"]:
                    compliance_stats[traffic_type]["throughput"] += 1
                if user_delay <= req["delay"]:
                    compliance_stats[traffic_type]["delay"] += 1
                if user_loss <= req["loss"]:
                    compliance_stats[traffic_type]["loss"] += 1

            # Save compliance stats
            for traffic_type in ["gbr", "non_gbr"]:
                if compliance_stats[traffic_type]["total"] > 0:
                    self.user_qos_stats[traffic_type].append(
                        {
                            "throughput_compliance": compliance_stats[traffic_type]["throughput"]
                            / compliance_stats[traffic_type]["total"]
                            * 100,
                            "delay_compliance": compliance_stats[traffic_type]["delay"]
                            / compliance_stats[traffic_type]["total"]
                            * 100,
                            "loss_compliance": compliance_stats[traffic_type]["loss"]
                            / compliance_stats[traffic_type]["total"]
                            * 100,
                        }
                    )

            # Save actual values
            for traffic_type in ["gbr", "non_gbr"]:
                self.user_actual_qos_values[traffic_type].append(
                    {
                        "avg_throughput": np.mean(actual_values[traffic_type]["throughput"]),
                        "avg_delay": np.mean(actual_values[traffic_type]["delay"]),
                        "avg_loss": np.mean(actual_values[traffic_type]["loss"]),
                        "median_throughput": np.median(actual_values[traffic_type]["throughput"]),
                        "median_delay": np.median(actual_values[traffic_type]["delay"]),
                        "median_loss": np.median(actual_values[traffic_type]["loss"]),
                        "min_throughput": np.min(actual_values[traffic_type]["throughput"]),
                        "max_delay": np.max(actual_values[traffic_type]["delay"]),
                        "max_loss": np.max(actual_values[traffic_type]["loss"]),
                    }
                )

            # Calculate overall compliance
            total_users = sum(stats["total"] for stats in compliance_stats.values())
            if total_users == 0:
                return 0.0

            qos_compliance = 0
            for traffic_type in compliance_stats:
                weight = compliance_stats[traffic_type]["total"] / total_users
                type_compliance = (
                    compliance_stats[traffic_type]["throughput"]
                    + compliance_stats[traffic_type]["delay"]
                    + compliance_stats[traffic_type]["loss"]
                ) / (3 * max(1, compliance_stats[traffic_type]["total"]))
                qos_compliance += type_compliance * weight

            return qos_compliance * 100
        except Exception as e:
            print(f"Error in QoS calculation: {e}")
            return 0.0

    def calculate_metrics(self, folder_path, params):
        """Вычисляет метрики для одной симуляции"""
        metrics = {}

        try:
            # Загружаем все необходимые файлы
            dl_mac = self.read_simulation_file(os.path.join(folder_path, "dl_mac_stats.csv"))
            dl_pdcp = self.read_simulation_file(os.path.join(folder_path, "dl_pdcp_stats.csv"))
            dl_tx_phy = self.read_simulation_file(os.path.join(folder_path, "dl_tx_phy_stats.csv"))

            if dl_pdcp is None or dl_mac is None:
                print(f"Skipping {folder_path} due to missing data")
                return None

            # Вычисляем время симуляции
            if "end" in dl_pdcp.columns and "start" in dl_pdcp.columns:
                simulation_time = dl_pdcp["end"].max() - dl_pdcp["start"].min()
            elif "% time" in dl_mac.columns:
                simulation_time = dl_mac["% time"].max() - dl_mac["% time"].min()
            else:
                simulation_time = 1.0

            if simulation_time <= 0:
                simulation_time = 1.0

            # 1. Пропускная способность
            total_tx_bytes = dl_pdcp["TxBytes"].sum() if "TxBytes" in dl_pdcp.columns else 0
            metrics["total_throughput"] = total_tx_bytes * 8 / simulation_time / 1e6  # Mbps

            # Средняя пропускная способность на пользователя
            ue_count = params.get(
                "ue_count", len(dl_pdcp["IMSI"].unique()) if "IMSI" in dl_pdcp.columns else 1
            )
            metrics["avg_user_throughput"] = (
                metrics["total_throughput"] / ue_count if ue_count > 0 else 0
            )

            # Пропускная способность по типам трафика (по IMSI)
            if "IMSI" in dl_pdcp.columns and "TxBytes" in dl_pdcp.columns:
                gbr_mask = dl_pdcp["IMSI"] % 2 == 0
                gbr_tx_bytes = (
                    dl_pdcp.loc[gbr_mask, "TxBytes"].sum() if "TxBytes" in dl_pdcp.columns else 0
                )
                non_gbr_tx_bytes = (
                    dl_pdcp.loc[~gbr_mask, "TxBytes"].sum() if "TxBytes" in dl_pdcp.columns else 0
                )

                metrics["gbr_throughput"] = gbr_tx_bytes * 8 / simulation_time / 1e6
                metrics["non_gbr_throughput"] = non_gbr_tx_bytes * 8 / simulation_time / 1e6
            else:
                metrics["gbr_throughput"] = 0
                metrics["non_gbr_throughput"] = 0

            # 2. Задержка и джиттер
            if "delay" in dl_pdcp.columns:
                delays = dl_pdcp["delay"].dropna()
                metrics["avg_delay"] = delays.mean() if len(delays) > 0 else 0

                # Джиттер (вариация задержки)
                metrics["jitter"] = delays.std() if len(delays) > 1 else 0
            else:
                metrics["avg_delay"] = 0
                metrics["jitter"] = 0

            # 3. Потеря пакетов
            if "nTxPDUs" in dl_pdcp.columns and "nRxPDUs" in dl_pdcp.columns:
                tx_packets = dl_pdcp["nTxPDUs"].sum()
                rx_packets = dl_pdcp["nRxPDUs"].sum()
                metrics["packet_loss"] = (
                    (tx_packets - rx_packets) / tx_packets * 100 if tx_packets > 0 else 0
                )
            else:
                metrics["packet_loss"] = 0

            # 4. Спектральная эффективность
            total_bits = 0
            if "sizeTb1" in dl_mac.columns:
                total_bits += dl_mac["sizeTb1"].sum() * 8
            if "sizeTb2" in dl_mac.columns:
                total_bits += dl_mac["sizeTb2"].sum() * 8

            bandwidth = params.get("bandwidth", 10)  # MHz
            metrics["spectral_efficiency"] = (
                total_bits / (bandwidth * 1e6 * simulation_time) if bandwidth > 0 else 0
            )

            # 5. Индекс справедливости (Jain's Fairness Index)
            if "IMSI" in dl_pdcp.columns and "TxBytes" in dl_pdcp.columns:
                throughput_per_ue = (
                    dl_pdcp.groupby("IMSI")["TxBytes"].sum() * 8 / simulation_time / 1e6
                )
                if len(throughput_per_ue) > 0:
                    sum_sq = np.sum(throughput_per_ue) ** 2
                    sum_of_sq = np.sum(throughput_per_ue**2)
                    metrics["fairness_index"] = (
                        sum_sq / (len(throughput_per_ue) * sum_of_sq) if sum_of_sq > 0 else 1
                    )
                else:
                    metrics["fairness_index"] = 1
            else:
                metrics["fairness_index"] = 1

            # 6. Эффективность планирования
            total_scheduled = len(dl_mac) if dl_mac is not None else 0
            max_possible_schedules = simulation_time * 1000  # предполагая 1 TTI = 1ms
            metrics["scheduling_efficiency"] = (
                total_scheduled / max_possible_schedules * 100 if max_possible_schedules > 0 else 0
            )

            # 7. Соответствие QoS требованиям
            metrics["qos_compliance"] = self.calculate_qos_compliance(dl_pdcp, ue_count)

            # Очистка памяти
            del dl_mac, dl_pdcp
            if dl_tx_phy is not None:
                del dl_tx_phy
            gc.collect()

            return metrics

        except Exception as e:
            print(f"Error processing {folder_path}: {str(e)}")
            return None

    def analyze_all_simulations(self):
        """Анализирует все папки с результатами симуляции"""
        folders = [
            f
            for f in os.listdir(self.results_dir)
            if os.path.isdir(os.path.join(self.results_dir, f))
        ]

        for folder in tqdm(folders, desc="Processing simulations"):
            folder_path = os.path.join(self.results_dir, folder)
            params = self.parse_folder_name(folder)
            metrics = self.calculate_metrics(folder_path, params)

            if metrics:
                self.simulation_params.append(params)
                for metric_name in self.metrics:
                    self.metrics[metric_name].append(metrics.get(metric_name, np.nan))

        # Создаем общий DataFrame для анализа
        self.results_df = pd.DataFrame(self.simulation_params)
        for metric_name in self.metrics:
            self.results_df[metric_name] = self.metrics[metric_name]

        # Заполняем отсутствующие параметры значениями по умолчанию
        if "bandwidth" not in self.results_df.columns:
            self.results_df["bandwidth"] = 10
        if "ue_count" not in self.results_df.columns:
            self.results_df["ue_count"] = 1
        if "distance" not in self.results_df.columns:
            self.results_df["distance"] = 100

    def save_plot_as_html(self, fig, filename):
        """Сохраняет график как HTML файл"""
        if not os.path.exists("plots"):
            os.makedirs("plots")
        fig.write_html(f"plots/{filename}.html")

    def plot_metrics_comparison(self, x_param="ue_count", hue_param="scheduler"):
        """Строит графики сравнения метрик по параметру с группировкой"""
        if not hasattr(self, "results_df"):
            self.analyze_all_simulations()

        # Проверяем наличие параметров в данных
        available_params = [
            col for col in self.results_df.columns if self.results_df[col].nunique() > 1
        ]

        if x_param not in available_params:
            print(
                f"Parameter {x_param} not found in results or has only one value. Available parameters: {available_params}"
            )
            x_param = available_params[0] if len(available_params) > 0 else "ue_count"

        if hue_param not in available_params:
            hue_param = None

        # Группируем метрики по категориям для лучшей визуализации
        metric_groups = {
            "Throughput Metrics": [
                "total_throughput",
                "avg_user_throughput",
                "gbr_throughput",
                "non_gbr_throughput",
            ],
            "Quality Metrics": [
                "avg_delay",
                "jitter",
                "packet_loss",
                "qos_compliance",
            ],
            "Efficiency Metrics": [
                "spectral_efficiency",
                "fairness_index",
                "scheduling_efficiency",
            ],
        }

        metric_titles = {
            "total_throughput": "Total Throughput (Mbps)",
            "avg_user_throughput": "Avg User Throughput (Mbps)",
            "gbr_throughput": "GBR Throughput (Mbps)",
            "non_gbr_throughput": "Non-GBR Throughput (Mbps)",
            "avg_delay": "Avg Packet Delay (ms)",
            "jitter": "Jitter (ms)",
            "packet_loss": "Packet Loss (%)",
            "spectral_efficiency": "Spectral Efficiency (bps/Hz)",
            "fairness_index": "Fairness Index (Jain's)",
            "scheduling_efficiency": "Scheduling Efficiency (%)",
            "qos_compliance": "QoS Compliance (%)",
        }

        # Создаем отдельные графики для каждой группы метрик
        for group_name, metrics in metric_groups.items():
            fig = make_subplots(
                rows=1,
                cols=len(metrics),
                subplot_titles=[metric_titles[m] for m in metrics],
                horizontal_spacing=0.1,
            )

            if hue_param:
                hue_values = sorted(self.results_df[hue_param].unique())
                colors = px.colors.qualitative.Plotly
            else:
                hue_values = [None]
                colors = ["blue"]

            sorted_df = self.results_df.sort_values(x_param)

            for i, metric in enumerate(metrics):
                for j, hue_value in enumerate(hue_values):
                    if hue_value is not None:
                        subset = sorted_df[sorted_df[hue_param] == hue_value]
                        name = str(hue_value)
                    else:
                        subset = sorted_df
                        name = "All"

                    # Создаем текст для отображения всех параметров в подсказке
                    hover_text = []
                    for idx, row in subset.iterrows():
                        param_text = "<br>".join(
                            [
                                f"{k}: {v}"
                                for k, v in row.items()
                                if k not in self.metrics and k != x_param
                            ]
                        )
                        hover_text.append(
                            f"{x_param}: {row[x_param]}<br>{metric_titles[metric]}: {row[metric]:.2f}<br>{param_text}"
                        )

                    fig.add_trace(
                        go.Scatter(
                            x=subset[x_param],
                            y=subset[metric],
                            name=name,
                            mode="lines+markers",
                            marker=dict(size=8, color=colors[j % len(colors)]),
                            line=dict(width=2, color=colors[j % len(colors)]),
                            legendgroup=name,
                            showlegend=(i == 0),
                            hovertemplate="%{text}<extra></extra>",
                            text=hover_text,
                        ),
                        row=1,
                        col=i + 1,
                    )

                fig.update_xaxes(
                    title_text=x_param.replace("_", " ").title(),
                    row=1,
                    col=i + 1,
                )
                fig.update_yaxes(
                    title_text=metric_titles[metric],
                    row=1,
                    col=i + 1,
                )

            fig.update_layout(
                title={
                    "text": f"<b>{group_name} vs {x_param.replace('_', ' ').title()}"
                    + (f" by {hue_param.replace('_', ' ').title()}" if hue_param else ""),
                    "y": 0.98,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                    "font": dict(size=16),
                },
                height=500,
                width=300 * len(metrics),
                margin=dict(t=100, b=80, l=80, r=80),
            )

            if hue_param:
                fig.update_layout(
                    legend=dict(
                        title=f"<b>{hue_param.replace('_', ' ').title()}</b>",
                        orientation="h",
                        yanchor="bottom",
                        y=1.1,
                        xanchor="right",
                        x=1,
                    )
                )

            # Сохраняем график как HTML
            filename = f"{group_name.lower().replace(' ', '_')}_vs_{x_param}"
            if hue_param:
                filename += f"_by_{hue_param}"
            self.save_plot_as_html(fig, filename)

            fig.show()

    def plot_qos_compliance_details(self):
        """Строит графики детализации соответствия QoS требованиям"""
        if not hasattr(self, "results_df"):
            self.analyze_all_simulations()

        if not self.user_qos_stats["gbr"] and not self.user_qos_stats["non_gbr"]:
            print("No QoS compliance data available")
            return

        # Create a figure with two rows: one for compliance, one for actual values
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "GBR Traffic QoS Compliance",
                "Non-GBR Traffic QoS Compliance",
                "GBR Traffic Actual Values",
                "Non-GBR Traffic Actual Values",
            ),
            horizontal_spacing=0.2,
            vertical_spacing=0.15,
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "box"}, {"type": "box"}]],
        )

        # Compliance plots (first row)
        for i, traffic_type in enumerate(["gbr", "non_gbr"]):
            stats = pd.DataFrame(self.user_qos_stats[traffic_type])
            if stats.empty:
                continue

            for j, metric in enumerate(
                ["throughput_compliance", "delay_compliance", "loss_compliance"]
            ):
                if metric in stats.columns:
                    fig.add_trace(
                        go.Bar(
                            x=[metric.replace("_", " ").title()],
                            y=[stats[metric].mean()],
                            name=traffic_type.upper(),
                            marker_color=px.colors.qualitative.Plotly[i * 3 + j],
                            showlegend=False,
                            text=[f"{stats[metric].mean():.1f}%"],
                            textposition="auto",
                        ),
                        row=1,
                        col=i + 1,
                    )

        # Actual values plots (second row)
        for i, traffic_type in enumerate(["gbr", "non_gbr"]):
            values = pd.DataFrame(self.user_actual_qos_values[traffic_type])
            if values.empty:
                continue

            # Throughput box plot
            fig.add_trace(
                go.Box(
                    y=values["avg_throughput"],
                    name="Throughput (Mbps)",
                    marker_color=px.colors.qualitative.Plotly[0],
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                    showlegend=(i == 0),
                ),
                row=2,
                col=i + 1,
            )

            # Delay box plot
            fig.add_trace(
                go.Box(
                    y=values["avg_delay"],
                    name="Delay (ms)",
                    marker_color=px.colors.qualitative.Plotly[1],
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                    showlegend=(i == 0),
                ),
                row=2,
                col=i + 1,
            )

            # Loss box plot
            fig.add_trace(
                go.Box(
                    y=values["avg_loss"],
                    name="Packet Loss (%)",
                    marker_color=px.colors.qualitative.Plotly[2],
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                    showlegend=(i == 0),
                ),
                row=2,
                col=i + 1,
            )

        # Add reference lines for QoS requirements
        qos_requirements = {
            "gbr": {"throughput": 5.0, "delay": 150.0, "loss": 0.1},
            "non_gbr": {"throughput": 1.0, "delay": 100.0, "loss": 0.1},
        }

        for i, traffic_type in enumerate(["gbr", "non_gbr"]):
            # Throughput reference line
            fig.add_hline(
                y=qos_requirements[traffic_type]["throughput"],
                line_dash="dot",
                line_color="green",
                annotation_text=f"Min Throughput: {qos_requirements[traffic_type]['throughput']}Mbps",
                annotation_position="top left",
                row=2,
                col=i + 1,
            )

            # Delay reference line
            fig.add_hline(
                y=qos_requirements[traffic_type]["delay"],
                line_dash="dot",
                line_color="red",
                annotation_text=f"Max Delay: {qos_requirements[traffic_type]['delay']}ms",
                annotation_position="top right",
                row=2,
                col=i + 1,
            )

            # Loss reference line
            fig.add_hline(
                y=qos_requirements[traffic_type]["loss"],
                line_dash="dot",
                line_color="orange",
                annotation_text=f"Max Loss: {qos_requirements[traffic_type]['loss']}%",
                annotation_position="bottom right",
                row=2,
                col=i + 1,
            )

        fig.update_layout(
            title="<b>QoS Compliance and Actual Performance Metrics</b>",
            height=800,
            width=1000,
            margin=dict(t=100, b=80, l=80, r=80),
            boxmode="group",
        )

        # Save and show the plot
        self.save_plot_as_html(fig, "qos_compliance_and_actual_values")
        fig.show()
        # else:
        #     print("No QoS compliance data to display")

    def plot_correlation_heatmap(self):
        """Строит тепловую карту корреляций между параметрами и метриками"""
        if not hasattr(self, "results_df"):
            self.analyze_all_simulations()

        numeric_cols = self.results_df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols = [
            col
            for col in numeric_cols
            if len(self.results_df[col].unique()) > 1 or col in self.metrics.keys()
        ]

        if not numeric_cols:
            print("No numeric columns found for correlation analysis")
            return

        corr_matrix = self.results_df[numeric_cols].corr()

        # Сортируем колонки для лучшего отображения
        metric_cols = [col for col in numeric_cols if col in self.metrics.keys()]
        param_cols = [col for col in numeric_cols if col not in self.metrics.keys()]
        sorted_cols = param_cols + metric_cols
        corr_matrix = corr_matrix.loc[sorted_cols, sorted_cols]

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Correlation"),
                hoverongaps=False,
                hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="<b>Correlation Between Simulation Parameters and QoS Metrics</b>",
            width=1000,
            height=1000,
            xaxis=dict(tickangle=45),
            margin=dict(l=150, r=50, b=150, t=100),
        )

        # Сохраняем график как HTML
        self.save_plot_as_html(fig, "correlation_heatmap")
        fig.show()


if __name__ == "__main__":
    analyzer = SimulationAnalyzer()
    analyzer.analyze_all_simulations()

    # Построение графиков сравнения
    analyzer.plot_metrics_comparison("ue_count")
    analyzer.plot_metrics_comparison("bandwidth")
    analyzer.plot_metrics_comparison("distance")

    # Детализация соответствия QoS
    analyzer.plot_qos_compliance_details()

    # Тепловая карта корреляций
    analyzer.plot_correlation_heatmap()

    # Сохранение результатов
    analyzer.results_df.to_csv("simulation_analysis_results.csv", index=False)

    if analyzer.user_qos_stats["gbr"]:
        pd.DataFrame(analyzer.user_qos_stats["gbr"]).to_csv("gbr_qos_stats.csv", index=False)
    if analyzer.user_qos_stats["non_gbr"]:
        pd.DataFrame(analyzer.user_qos_stats["non_gbr"]).to_csv(
            "non_gbr_qos_stats.csv", index=False
        )
    # df = analyzer.read_simulation_file(
    #     "SimulationResults/Urban_BW100_UEs1_TM3_CAOn_SchedPF_MobPedestrian_GBRvideo_NonGBRvoice/dl_pdcp_stats.csv"
    # )
    # print(df["TxBytes"].sum())
    # ,
    # 10,
    # )
    # )
