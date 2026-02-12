"""
PLOTTING UTILITIES
==================

Engineering-relevant visualizations for bending analysis
"""

import plotly.graph_objects as go
import numpy as np


class BendingVisualizer:
    """
    Create clean, deterministic engineering plots
    """
    
    def __init__(self):
        self.colors = {
            'neutral': '#95a5a6',
            'overbend': '#e74c3c',
            'final': '#3498db',
            'target': '#2ecc71'
        }
        
    def plot_bending_curves(self, target_angle, overbend_angle, final_angle):
        """
        Visualize: Neutral → Overbend → Final bend
        
        Returns: plotly figure
        """
        # Simplified bend profiles (straight → curved → relaxed)
        x = np.linspace(0, 100, 100)
        
        # Neutral plate (straight)
        z_neutral = np.zeros_like(x)
        
        # Overbend curve (maximum deformation)
        z_overbend = self._generate_bend_curve(x, overbend_angle)
        
        # Final curve (after springback)
        z_final = self._generate_bend_curve(x, final_angle)
        
        # Create figure
        fig = go.Figure()
        
        # Neutral
        fig.add_trace(go.Scatter(
            x=x, y=z_neutral,
            mode='lines',
            name='Initial Plate (Neutral)',
            line=dict(color=self.colors['neutral'], width=2, dash='dash'),
            hovertemplate='<b>Neutral</b><br>X: %{x:.1f} mm<br>Z: %{y:.2f} mm<extra></extra>'
        ))
        
        # Overbend
        fig.add_trace(go.Scatter(
            x=x, y=z_overbend,
            mode='lines',
            name=f'Overbend ({overbend_angle:.1f}°)',
            line=dict(color=self.colors['overbend'], width=3),
            hovertemplate='<b>Overbend</b><br>X: %{x:.1f} mm<br>Z: %{y:.2f} mm<extra></extra>'
        ))
        
        # Final after springback
        fig.add_trace(go.Scatter(
            x=x, y=z_final,
            mode='lines',
            name=f'Final After Springback ({final_angle:.1f}°)',
            line=dict(color=self.colors['final'], width=3),
            hovertemplate='<b>Final</b><br>X: %{x:.1f} mm<br>Z: %{y:.2f} mm<extra></extra>'
        ))
        
        # Layout
        fig.update_layout(
            title={
                'text': 'Bending Profile: Overbend → Springback → Final',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Arial, sans-serif'}
            },
            xaxis_title=dict(text='Distance along sheet (mm)', font=dict(size=14)),
            yaxis_title=dict(text='Out-of-plane deflection (mm)', font=dict(size=14)),
            template='plotly_white',
            height=500,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                font=dict(size=12)
            ),
            margin=dict(l=60, r=40, t=80, b=60)
        )
        
        return fig
    
    def plot_angle_comparison(self, target, overbend, final, Ks):
        """
        Bar chart: Target vs Overbend vs Final
        
        Returns: plotly figure
        """
        fig = go.Figure()
        
        angles = ['Target', 'Overbend\n(Commanded)', 'Final\n(Predicted)']
        values = [target, overbend, final]
        colors_list = [self.colors['target'], self.colors['overbend'], self.colors['final']]
        
        # Determine max value for Y-axis scaling
        max_val = max(values)
        
        fig.add_trace(go.Bar(
            x=angles,
            y=values,
            marker_color=colors_list,
            text=[f'{v:.1f}°' for v in values],
            textposition='outside',
            textfont=dict(size=14),
            hovertemplate='<b>%{x}</b><br>Angle: %{y:.2f}°<extra></extra>'
        ))
        
        # Add springback annotation
        fig.add_annotation(
            x=1, y=overbend,
            text=f'Springback<br>Factor: {Ks:.3f}',
            showarrow=True,
            arrowhead=2,
            ax=50,
            ay=-40,
            font=dict(size=12, color='#e74c3c'),
            bgcolor='white',
            bordercolor='#e74c3c',
            borderwidth=1
        )
        
        fig.update_layout(
            title={
                'text': 'Angle Comparison with Springback Compensation',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            yaxis=dict(
                title=dict(text='Bend Angle (degrees)', font=dict(size=14)),
                range=[0, max_val * 1.3]  # Add 30% headroom for text labels
            ),
            xaxis=dict(
                tickfont=dict(size=12)
            ),
            template='plotly_white',
            height=450,
            showlegend=False,
            margin=dict(l=60, r=40, t=100, b=60)  # Increased top margin
        )
        
        return fig
    
    def _generate_bend_curve(self, x, angle_degrees):
        """
        Generate simplified bend curve for visualization
        
        Simplified arc approximation for display purposes
        """
        # Convert angle to radians
        angle_rad = np.deg2rad(angle_degrees)
        
        # Bend radius (larger angle = tighter bend)
        R = 100 / (angle_rad + 0.01)  # Simplified relationship
        
        # Circular arc approximation
        z = R * (1 - np.cos(angle_rad * x / 100)) if angle_degrees > 0 else np.zeros_like(x)
        
        return z
    
    def plot_toolpath_preview(self, toolpath_df):
        """
        Plot toolpath parameters over steps
        
        Args:
            toolpath_df: DataFrame from ToolpathGenerator
            
        Returns: plotly figure
        """
        fig = go.Figure()
        
        # RX trajectory
        fig.add_trace(go.Scatter(
            x=toolpath_df['step_id'],
            y=toolpath_df['rx'],
            mode='lines+markers',
            name='RX (Tool Rotation)',
            line=dict(color='#3498db', width=2),
            marker=dict(size=4)
        ))
        
        # RYR trajectory
        fig.add_trace(go.Scatter(
            x=toolpath_df['step_id'],
            y=toolpath_df['ryr'],
            mode='lines+markers',
            name='RYR (Constraint)',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=4),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Toolpath Parameter Trajectories',
            xaxis_title='Step ID',
            yaxis_title='RX (degrees)',
            yaxis2=dict(
                title='RYR',
                overlaying='y',
                side='right'
            ),
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )
        
        return fig


# Quick test
if __name__ == "__main__":
    viz = BendingVisualizer()
    
    # Test bend curves
    fig1 = viz.plot_bending_curves(
        target_angle=45,
        overbend_angle=48,
        final_angle=45
    )
    
    # Test angle comparison
    fig2 = viz.plot_angle_comparison(
        target=45,
        overbend=48,
        final=45,
        Ks=0.9375
    )
    
    print("✓ Visualization utilities ready")
