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
            # Throughput metrics
            "total_throughput": [],
            "avg_user_throughput": [],
            "gbr_throughput": [],
            "non_gbr_throughput": [],
            "voice_throughput": [],
            "video_throughput": [],
            "gaming_throughput": [],
            "best_effort_throughput": [],
            
            # Quality metrics
            "avg_delay": [],
            "jitter": [],
            "packet_loss": [],
            
            # Efficiency metrics
            "spectral_efficiency": [],
            "fairness_index": [],
            
            # QoS compliance metrics
            "qos_compliance": [],
            "voice_throughput_compliance": [],
            "voice_delay_compliance": [],
            "voice_loss_compliance": [],
            "video_throughput_compliance": [],
            "video_delay_compliance": [],
            "video_loss_compliance": [],
            "gaming_throughput_compliance": [],
            "gaming_delay_compliance": [],
            "gaming_loss_compliance": [],
            "best_effort_throughput_compliance": [],
            "best_effort_delay_compliance": [],
            "best_effort_loss_compliance": [],
            
            # Actual QoS values
            "voice_avg_throughput": [],
            "voice_avg_delay": [],
            "voice_avg_loss": [],
            "video_avg_throughput": [],
            "video_avg_delay": [],
            "video_avg_loss": [],
            "gaming_avg_throughput": [],
            "gaming_avg_delay": [],
            "gaming_avg_loss": [],
            "best_effort_avg_throughput": [],
            "best_effort_avg_delay": [],
            "best_effort_avg_loss": [],
        }
        self.simulation_params = []

    def parse_folder_name(self, folder_name):
        """Extract simulation parameters from folder name"""
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
        }

        parts = params_parts[0].split("_")

        for part in parts:
            for key in param_mapping:
                if part.startswith(key):
                    param_name = param_mapping[key]
                    param_value = part[len(key):]

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
        """Read simulation results file"""
        try:
            try:
                df = pd.read_csv(filepath, sep="\t", index_col=False)
            except:
                df = pd.read_csv(filepath)

            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None

    def calculate_qos_compliance(self, dl_pdcp, ue_count):
        """Calculate QoS compliance and actual values for each traffic type"""
        if dl_pdcp is None or "IMSI" not in dl_pdcp.columns:
            return 0.0, {}

        qos_requirements = {
            "voice": {"throughput": 0.3, "delay": 0.01, "loss": 0.01},
            "video": {"throughput": 7.5, "delay": 0.015, "loss": 0.001},
            "gaming": {"throughput": 2.5, "delay": 0.005, "loss": 0.001},
            "best_effort": {"throughput": 0.3, "delay": 0.03, "loss": 0.01},
        }

        compliance_stats = {k: {"throughput": 0, "delay": 0, "loss": 0, "total": 0} for k in qos_requirements}
        actual_values = {k: {"throughput": [], "delay": [], "loss": []} for k in qos_requirements}

        try:
            grouped = dl_pdcp.groupby("IMSI")

            for imsi, group in grouped:
                i = int(imsi) - 1
                traffic_type = ["voice", "video", "gaming", "best_effort"][i % 4]
                req = qos_requirements[traffic_type]
                compliance_stats[traffic_type]["total"] += 1

                time_diff = group["end"].max() - group["% start"].min()
                time_diff = time_diff if time_diff > 0 else 1.0

                user_throughput = group["TxBytes"].sum() * 8 / time_diff / 1e6
                user_delay = group["delay"].mean() if "delay" in group.columns else 0

                tx_packets = group["TxBytes"].sum()
                rx_packets = group["RxBytes"].sum() if "RxBytes" in group.columns else tx_packets
                user_loss = (tx_packets - rx_packets) / tx_packets * 100 if tx_packets > 0 else 0

                actual_values[traffic_type]["throughput"].append(user_throughput)
                actual_values[traffic_type]["delay"].append(user_delay)
                actual_values[traffic_type]["loss"].append(user_loss)

                if user_throughput >= req["throughput"]:
                    compliance_stats[traffic_type]["throughput"] += 1
                if user_delay <= req["delay"]:
                    compliance_stats[traffic_type]["delay"] += 1
                if user_loss <= req["loss"]:
                    compliance_stats[traffic_type]["loss"] += 1

            per_type_compliance = {}
            for t in qos_requirements:
                total = compliance_stats[t]["total"]
                if total > 0:
                    throughput_p = 100 * compliance_stats[t]["throughput"] / total
                    delay_p = 100 * compliance_stats[t]["delay"] / total
                    loss_p = 100 * compliance_stats[t]["loss"] / total
                    per_type_compliance[t] = {
                        "throughput_compliance": throughput_p,
                        "delay_compliance": delay_p,
                        "loss_compliance": loss_p,
                        "avg_throughput": np.mean(actual_values[t]["throughput"]),
                        "avg_delay": np.mean(actual_values[t]["delay"]),
                        "avg_loss": np.mean(actual_values[t]["loss"]),
                        "median_throughput": np.median(actual_values[t]["throughput"]),
                        "median_delay": np.median(actual_values[t]["delay"]),
                        "median_loss": np.median(actual_values[t]["loss"]),
                        "min_throughput": np.min(actual_values[t]["throughput"]),
                        "max_delay": np.max(actual_values[t]["delay"]),
                        "max_loss": np.max(actual_values[t]["loss"]),
                    }

            total_users = sum(v["total"] for v in compliance_stats.values())
            if total_users == 0:
                return 0.0, per_type_compliance

            qos_compliance = 0.0
            for t in qos_requirements:
                weight = compliance_stats[t]["total"] / total_users
                per_type = (
                    compliance_stats[t]["throughput"]
                    + compliance_stats[t]["delay"]
                    + compliance_stats[t]["loss"]
                ) / (3 * max(1, compliance_stats[t]["total"]))
                qos_compliance += per_type * weight

            return qos_compliance * 100, per_type_compliance
        except Exception as e:
            print(f"Error in QoS calculation: {e}")
            return 0.0, {}

    def compute_throughput_by_traffic_type(self, df, simulation_time):
        qos_traffic_map = {
            0: "voice",
            1: "video",
            2: "gaming",
            3: "best_effort",
        }

        per_type_throughput = defaultdict(float)

        if "IMSI" not in df.columns or "TxBytes" not in df.columns:
            return per_type_throughput

        grouped = df.groupby("IMSI")
        for imsi, group in grouped:
            traffic_type = qos_traffic_map[(int(imsi) -1) % 4]
            tx_bytes = group["TxBytes"].sum()
            per_type_throughput[traffic_type] += tx_bytes * 8 / simulation_time / 1e6  # Mbps

        per_type_throughput["gbr_total"] = (
            per_type_throughput["voice"]
            + per_type_throughput["video"]
            + per_type_throughput["gaming"]
        )
        per_type_throughput["non_gbr_total"] = per_type_throughput["best_effort"]

        return per_type_throughput

    def calculate_metrics(self, folder_path, params):
        """Calculate metrics for a single simulation"""
        metrics = {}

        try:
            dl_mac = self.read_simulation_file(os.path.join(folder_path, "dl_mac_stats.csv"))
            dl_pdcp = self.read_simulation_file(os.path.join(folder_path, "dl_pdcp_stats.csv"))

            if dl_pdcp is None or dl_mac is None:
                print(f"Skipping {folder_path} due to missing data")
                return None

            if "end" in dl_pdcp.columns and "% start" in dl_pdcp.columns:
                simulation_time = dl_pdcp["end"].max() - dl_pdcp["% start"].min()
            else:
                simulation_time = 1.0

            if simulation_time <= 0:
                simulation_time = 1.0
            
            per_type_tp = self.compute_throughput_by_traffic_type(dl_pdcp, simulation_time)
            for ttype, value in per_type_tp.items():
                metrics[f"{ttype}_throughput"] = value

            # 1. Throughput metrics
            total_tx_bytes = dl_pdcp["TxBytes"].sum() if "TxBytes" in dl_pdcp.columns else 0
            metrics["total_throughput"] = total_tx_bytes * 8 / simulation_time / 1e6  # Mbps

            ue_count = params.get(
                "ue_count", len(dl_pdcp["IMSI"].unique()) if "IMSI" in dl_pdcp.columns else 1
            )
            metrics["avg_user_throughput"] = (
                metrics["total_throughput"] / ue_count if ue_count > 0 else 0
            )

            # GBR/Non-GBR throughput
            if "IMSI" in dl_pdcp.columns and "TxBytes" in dl_pdcp.columns:
                ngbr_mask = dl_pdcp["IMSI"] % 4 == 0
                gbr_tx_bytes = (
                    dl_pdcp.loc[~ngbr_mask, "TxBytes"].sum() if "TxBytes" in dl_pdcp.columns else 0
                )
                non_gbr_tx_bytes = (
                    dl_pdcp.loc[ngbr_mask, "TxBytes"].sum() if "TxBytes" in dl_pdcp.columns else 0
                )

                metrics["gbr_throughput"] = gbr_tx_bytes * 8 / simulation_time / 1e6
                metrics["non_gbr_throughput"] = non_gbr_tx_bytes * 8 / simulation_time / 1e6
            else:
                metrics["gbr_throughput"] = 0
                metrics["non_gbr_throughput"] = 0

            # 2. Delay and jitter
            if "delay" in dl_pdcp.columns:
                delays = dl_pdcp["delay"].dropna()
                metrics["avg_delay"] = delays.mean() if len(delays) > 0 else 1
                jitter = dl_pdcp["stdDev"].dropna()
                metrics["jitter"] = jitter.mean() if len(jitter) > 0 else 1
            else:
                metrics["avg_delay"] = 1
                metrics["jitter"] = 1

            # 3. Packet loss
            if "nTxPDUs" in dl_pdcp.columns and "nRxPDUs" in dl_pdcp.columns:
                tx_packets = dl_pdcp["nTxPDUs"].sum()
                rx_packets = dl_pdcp["nRxPDUs"].sum()
                metrics["packet_loss"] = (
                    (tx_packets - rx_packets) / tx_packets * 100 if tx_packets > 0 else 0
                )
            else:
                metrics["packet_loss"] = 0

            # 4. Spectral efficiency
            total_bits = 0
            if "sizeTb1" in dl_mac.columns:
                total_bits += dl_mac["sizeTb1"].sum() * 8
            if "sizeTb2" in dl_mac.columns:
                total_bits += dl_mac["sizeTb2"].sum() * 8

            bandwidth = params.get("bandwidth", 10)  # MHz
            metrics["spectral_efficiency"] = (
                total_bits / (bandwidth * 1e6 * simulation_time) if bandwidth > 0 else 0
            )

            # 5. Fairness index
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
                metrics["fairness_index"] = 0

            # 6. QoS compliance
            qos_compliance, per_type_qos = self.calculate_qos_compliance(dl_pdcp, ue_count)
            metrics["qos_compliance"] = qos_compliance
            
            for ttype, values in per_type_qos.items():
                for k, v in values.items():
                    metrics[f"{ttype}_{k}"] = v

            del dl_mac, dl_pdcp
            gc.collect()

            return metrics

        except Exception as e:
            print(f"Error processing {folder_path}: {str(e)}")
            return None

    def analyze_all_simulations(self):
        """Analyze all simulation folders"""
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

        self.results_df = pd.DataFrame(self.simulation_params)
        for metric_name in self.metrics:
            self.results_df[metric_name] = self.metrics[metric_name]

        if "bandwidth" not in self.results_df.columns:
            self.results_df["bandwidth"] = 10
        if "ue_count" not in self.results_df.columns:
            self.results_df["ue_count"] = 1
        if "distance" not in self.results_df.columns:
            self.results_df["distance"] = 100

    def save_plot_as_html(self, fig, filename):
        """Save plot as HTML file"""
        if not os.path.exists("plots"):
            os.makedirs("plots")
        fig.write_html(f"plots/{filename}.html")

    def plot_metrics_comparison(self, x_param="ue_count", hue_param="scheduler"):
        """Plot metrics comparison by parameter with grouping and up to 4 subplots per figure."""
        if not hasattr(self, "results_df"):
            self.analyze_all_simulations()

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

        metric_groups = {
            "Throughput Metrics": [
                "total_throughput", "avg_user_throughput", "gbr_throughput", "non_gbr_throughput",
                "voice_throughput", "video_throughput", "gaming_throughput", "best_effort_throughput",
            ],
            "Quality Metrics": [
                "avg_delay", "jitter", "packet_loss",
                "voice_avg_delay", "video_avg_delay", "gaming_avg_delay", "best_effort_avg_delay",
                "voice_avg_loss", "video_avg_loss", "gaming_avg_loss", "best_effort_avg_loss",
            ],
            "QoS Compliance Metrics": [
                "qos_compliance",
                "voice_throughput_compliance", "voice_delay_compliance", "voice_loss_compliance",
                "video_throughput_compliance", "video_delay_compliance", "video_loss_compliance",
                "gaming_throughput_compliance", "gaming_delay_compliance", "gaming_loss_compliance",
                "best_effort_throughput_compliance", "best_effort_delay_compliance", "best_effort_loss_compliance",
            ],
            "Efficiency Metrics": [
                "spectral_efficiency", "fairness_index",
            ],
        }

        metric_titles = {
            "total_throughput": "Total Throughput (Mbps)",
            "avg_user_throughput": "Avg User Throughput (Mbps)",
            "gbr_throughput": "GBR Throughput (Mbps)",
            "non_gbr_throughput": "Non-GBR Throughput (Mbps)",
            "voice_throughput": "Voice Throughput (Mbps)",
            "video_throughput": "Video Throughput (Mbps)",
            "gaming_throughput": "Gaming Throughput (Mbps)",
            "best_effort_throughput": "Best Effort Throughput (Mbps)",

            "avg_delay": "Avg Packet Delay (ms)",
            "jitter": "Jitter (ms)",
            "packet_loss": "Packet Loss (%)",
            "voice_avg_delay": "Voice Avg Delay (ms)",
            "video_avg_delay": "Video Avg Delay (ms)",
            "gaming_avg_delay": "Gaming Avg Delay (ms)",
            "best_effort_avg_delay": "Best Effort Avg Delay (ms)",
            "voice_avg_loss": "Voice Avg Loss (%)",
            "video_avg_loss": "Video Avg Loss (%)",
            "gaming_avg_loss": "Gaming Avg Loss (%)",
            "best_effort_avg_loss": "Best Effort Avg Loss (%)",

            "qos_compliance": "Overall QoS Compliance (%)",
            "voice_throughput_compliance": "Voice Throughput Compliance (%)",
            "voice_delay_compliance": "Voice Delay Compliance (%)",
            "voice_loss_compliance": "Voice Loss Compliance (%)",
            "video_throughput_compliance": "Video Throughput Compliance (%)",
            "video_delay_compliance": "Video Delay Compliance (%)",
            "video_loss_compliance": "Video Loss Compliance (%)",
            "gaming_throughput_compliance": "Gaming Throughput Compliance (%)",
            "gaming_delay_compliance": "Gaming Delay Compliance (%)",
            "gaming_loss_compliance": "Gaming Loss Compliance (%)",
            "best_effort_throughput_compliance": "Best Effort Throughput Compliance (%)",
            "best_effort_delay_compliance": "Best Effort Delay Compliance (%)",
            "best_effort_loss_compliance": "Best Effort Loss Compliance (%)",

            "spectral_efficiency": "Spectral Efficiency (bps/Hz)",
            "fairness_index": "Fairness Index (Jain's)",
        }

        if hue_param:
            hue_values = sorted(self.results_df[hue_param].unique())
            colors = px.colors.qualitative.Plotly
        else:
            hue_values = [None]
            colors = ["blue"]

        sorted_df = self.results_df.sort_values(x_param)

        for group_name, metrics in metric_groups.items():
            existing_metrics = [m for m in metrics if m in self.results_df.columns]
            if not existing_metrics:
                continue

            for chunk_idx in range(0, len(existing_metrics), 4):
                chunk_metrics = existing_metrics[chunk_idx:chunk_idx+4]
                
                fig = make_subplots(
                    rows=1, cols=len(chunk_metrics),
                    subplot_titles=[metric_titles[m] for m in chunk_metrics],
                    horizontal_spacing=0.08,
                )

                for i, metric in enumerate(chunk_metrics):
                    for j, hue_value in enumerate(hue_values):
                        if hue_value is not None:
                            subset = sorted_df[sorted_df[hue_param] == hue_value]
                            name = str(hue_value)
                        else:
                            subset = sorted_df
                            name = "All"

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
                            row=1, col=i+1,
                        )

                    fig.update_xaxes(title_text=x_param.replace("_", " ").title(), row=1, col=i+1)
                    fig.update_yaxes(title_text=metric_titles[metric], row=1, col=i+1)

                fig.update_layout(
                    title={
                        "text": f"<b>{group_name} (Part {chunk_idx//4 + 1}) vs {x_param.replace('_', ' ').title()}" +
                                (f" by {hue_param.replace('_', ' ').title()}" if hue_param else ""),
                        "y": 0.98,
                        "x": 0.5,
                        "xanchor": "center",
                        "yanchor": "top",
                        "font": dict(size=16),
                    },
                    height=500,
                    width=300 * len(chunk_metrics),
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

                filename = f"{group_name.lower().replace(' ', '_')}_part_{chunk_idx//4 + 1}_vs_{x_param}"
                if hue_param:
                    filename += f"_by_{hue_param}"
                self.save_plot_as_html(fig, filename)

                fig.show()


    def plot_qos_compliance_details(self, x_param="ue_count", hue_param="scheduler"):
        """Plot detailed QoS compliance metrics separately for each metric"""
        if not hasattr(self, "results_df"):
            self.analyze_all_simulations()

        traffic_types = ["voice", "video", "gaming", "best_effort"]
        metrics = ["throughput_compliance", "delay_compliance", "loss_compliance"]

        if hue_param:
            hue_values = sorted(self.results_df[hue_param].unique())
            colors = px.colors.qualitative.Plotly
        else:
            hue_values = [None]
            colors = ["blue"]

        sorted_df = self.results_df.sort_values(x_param)

        for metric in metrics:
            fig = go.Figure()
            for traffic_type in traffic_types:
                col_name = f"{traffic_type}_{metric}"
                if col_name not in self.results_df.columns:
                    continue

                for k, hue_value in enumerate(hue_values):
                    if hue_value is not None:
                        subset = sorted_df[sorted_df[hue_param] == hue_value]
                        name = f"{traffic_type.title()} - {hue_value}"
                    else:
                        subset = sorted_df
                        name = f"{traffic_type.title()}"

                    fig.add_trace(
                        go.Scatter(
                            x=subset[x_param],
                            y=subset[col_name],
                            mode="lines+markers",
                            name=name,
                            marker=dict(size=8, color=colors[k % len(colors)]),
                            line=dict(width=2, color=colors[k % len(colors)]),
                            legendgroup=str(hue_value) if hue_value else traffic_type,
                            hovertemplate=f"{x_param}: %{{x}}<br>{traffic_type.title()} {metric.replace('_', ' ')}: %{{y:.1f}}%<extra></extra>"
                        )
                    )

            fig.update_layout(
                title=f"<b>{metric.replace('_', ' ').title()} vs {x_param.replace('_', ' ').title()}" +
                    (f" by {hue_param.replace('_', ' ').title()}" if hue_param else ""),
                xaxis_title=x_param.replace("_", " ").title(),
                yaxis_title=f"{metric.replace('_', ' ').title()} (%)",
                height=600,
                width=900,
                legend_title=hue_param.replace("_", " ").title() if hue_param else None,
                margin=dict(t=80, b=80, l=80, r=80)
            )

            filename = f"qos_compliance_{metric}_vs_{x_param}"
            if hue_param:
                filename += f"_by_{hue_param}"

            self.save_plot_as_html(fig, filename)
            fig.show()


    def plot_actual_qos_values(self, x_param="ue_count", hue_param="scheduler"):
        """Plot actual QoS values separately for each metric"""
        if not hasattr(self, "results_df"):
            self.analyze_all_simulations()

        traffic_types = ["voice", "video", "gaming", "best_effort"]
        metrics = ["avg_throughput", "avg_delay", "avg_loss"]

        qos_requirements = {
            "voice": {"avg_throughput": 0.3, "avg_delay": 100.0, "avg_loss": 0.1},
            "video": {"avg_throughput": 7.5, "avg_delay": 150.0, "avg_loss": 0.1},
            "gaming": {"avg_throughput": 2.5, "avg_delay": 100.0, "avg_loss": 0.1},
            "best_effort": {"avg_throughput": 1.0, "avg_delay": 300.0, "avg_loss": 1.0}
        }

        if hue_param:
            hue_values = sorted(self.results_df[hue_param].unique())
            colors = px.colors.qualitative.Plotly
        else:
            hue_values = [None]
            colors = ["blue"]

        sorted_df = self.results_df.sort_values(x_param)

        for metric in metrics:
            fig = go.Figure()
            for traffic_type in traffic_types:
                col_name = f"{traffic_type}_{metric}"
                if col_name not in self.results_df.columns:
                    continue

                for k, hue_value in enumerate(hue_values):
                    if hue_value is not None:
                        subset = sorted_df[sorted_df[hue_param] == hue_value]
                        name = f"{traffic_type.title()} - {hue_value}"
                    else:
                        subset = sorted_df
                        name = f"{traffic_type.title()}"

                    fig.add_trace(
                        go.Scatter(
                            x=subset[x_param],
                            y=subset[col_name],
                            mode="lines+markers",
                            name=name,
                            marker=dict(size=8, color=colors[k % len(colors)]),
                            line=dict(width=2, color=colors[k % len(colors)]),
                            legendgroup=str(hue_value) if hue_value else traffic_type,
                            hovertemplate=f"{x_param}: %{{x}}<br>{metric.replace('_', ' ')}: %{{y:.2f}}" +
                                        (" Mbps" if "throughput" in metric else " ms" if "delay" in metric else " %") + "<extra></extra>"
                        )
                    )

                for traffic_type in traffic_types:
                    req_value = qos_requirements[traffic_type].get(metric)
                    if req_value is not None:
                        fig.add_hline(
                            y=req_value,
                            line_dash="dot",
                            line_color="green" if "throughput" in metric else "red" if "delay" in metric else "orange",
                            annotation_text=f"{traffic_type.title()} QoS {'Min' if 'throughput' in metric else 'Max'}",
                            annotation_position="top right" if "delay" in metric else "bottom right",
                            annotation_font_size=10
                        )

            yaxis_titles = {
                "avg_throughput": "Throughput (Mbps)",
                "avg_delay": "Delay (ms)",
                "avg_loss": "Loss (%)"
            }

            fig.update_layout(
                title=f"<b>{metric.replace('_', ' ').title()} vs {x_param.replace('_', ' ').title()}" +
                    (f" by {hue_param.replace('_', ' ').title()}" if hue_param else ""),
                xaxis_title=x_param.replace("_", " ").title(),
                yaxis_title=yaxis_titles.get(metric, metric),
                height=600,
                width=900,
                legend_title=hue_param.replace("_", " ").title() if hue_param else None,
                margin=dict(t=80, b=80, l=80, r=80)
            )

            filename = f"actual_qos_values_{metric}_vs_{x_param}"
            if hue_param:
                filename += f"_by_{hue_param}"

            self.save_plot_as_html(fig, filename)
            fig.show()


    def plot_correlation_heatmap(self):
        """Plot correlation heatmap between parameters and metrics"""
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

        self.save_plot_as_html(fig, "correlation_heatmap")
        fig.show()

if __name__ == "__main__":
    analyzer = SimulationAnalyzer()
    analyzer.analyze_all_simulations()

    analyzer.plot_metrics_comparison("ue_count")
    analyzer.plot_metrics_comparison("bandwidth")
    analyzer.plot_metrics_comparison("distance")

    analyzer.plot_qos_compliance_details()

    analyzer.plot_correlation_heatmap()

    analyzer.results_df.to_csv("simulation_analysis_results.csv", index=False)
