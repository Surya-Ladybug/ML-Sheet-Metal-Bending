"""
TOOLPATH GENERATION MODULE
===========================

Purpose:
    Generate robot bending toolpath from ML predictions
    
Assumptions:
    - Rotary bending mechanism (single-axis rotation)
    - Constant angular velocity during bend
    - Small-angle discretization for smooth motion
    - Linear interpolation between key points
    
Output:
    Toolpath table with step-by-step commands for robot controller
"""

import pandas as pd
import numpy as np


class ToolpathGenerator:
    """
    Generate robot toolpath from ML-predicted bending parameters
    """
    
    def __init__(self, angular_step=1.0):
        """
        Initialize toolpath generator
        
        Args:
            angular_step: Discretization step size (degrees)
        """
        self.angular_step = angular_step
        
    def generate_toolpath(self, 
                         rx_start, rx_end,
                         ryr_start, ryr_end,
                         overbend_angle,
                         final_angle,
                         n_steps=None):
        """
        Generate complete bending toolpath
        
        Engineering Logic:
        1. Start from neutral position (rx=0, ryr=0)
        2. Ramp to working parameters
        3. Execute overbend (to compensate springback)
        4. Release to final angle after springback
        
        Args:
            rx_start: Initial rx (usually 0)
            rx_end: Target rx from inverse model
            ryr_start: Initial ryr (usually 0)
            ryr_end: Target ryr from inverse model
            overbend_angle: Angle with springback compensation
            final_angle: Expected final angle after relaxation
            n_steps: Number of steps (auto if None)
            
        Returns:
            pd.DataFrame: Toolpath with columns:
                - step_id
                - rx
                - ryr
                - tool_angle
                - predicted_angle
                - springback_corrected_angle
        """
        
        # Auto-calculate steps if not provided
        if n_steps is None:
            angle_range = abs(rx_end - rx_start)
            n_steps = max(int(angle_range / self.angular_step), 10)
        
        # Generate interpolated path
        steps = np.linspace(0, n_steps-1, n_steps, dtype=int)
        
        # Linear interpolation for rx and ryr
        rx_path = np.linspace(rx_start, rx_end, n_steps)
        ryr_path = np.linspace(ryr_start, ryr_end, n_steps)
        
        # Tool angle (physical rotation)
        tool_angle_path = np.linspace(0, abs(rx_end), n_steps)
        
        # Predicted angle (from forward model)
        predicted_angle_path = np.linspace(0, final_angle, n_steps)
        
        # Springback-corrected angle (overbend during process)
        springback_corrected_path = np.linspace(0, overbend_angle, n_steps)
        
        # Build toolpath DataFrame
        toolpath = pd.DataFrame({
            'step_id': steps,
            'rx': rx_path,
            'ryr': ryr_path,
            'tool_angle': tool_angle_path,
            'predicted_angle': predicted_angle_path,
            'springback_corrected_angle': springback_corrected_path
        })
        
        return toolpath
    
    def generate_simple_toolpath(self, rx, ryr, overbend_angle, final_angle):
        """
        Simplified toolpath for UI display
        
        Returns: DataFrame with fewer columns for clarity
        """
        n_steps = max(int(abs(rx) / self.angular_step), 10)
        
        toolpath = pd.DataFrame({
            'Step': range(1, n_steps + 1),
            'RX (°)': np.linspace(0, rx, n_steps),
            'RYR': np.linspace(0, ryr, n_steps),
            'Tool Angle (°)': np.linspace(0, abs(rx), n_steps),
            'Target Angle (°)': [final_angle] * n_steps,
            'Overbend (°)': [overbend_angle] * n_steps
        })
        
        return toolpath
    
    def export_toolpath(self, toolpath, filepath):
        """
        Export toolpath to CSV for robot controller
        
        Format: Industrial robot-ready format
        """
        toolpath.to_csv(filepath, index=False)
        print(f"✓ Toolpath exported: {filepath}")
        
        return filepath
    
    def generate_summary(self, toolpath, rx, ryr, final_angle, overbend_angle, Ks):
        """
        Generate human-readable toolpath summary
        
        Returns: Formatted string for UI or report
        """
        summary = f"""
TOOLPATH GENERATION SUMMARY
{'='*60}

Toolpath Parameters:
  Total Steps:        {len(toolpath)}
  Angular Step Size:  {self.angular_step}°
  
Input Parameters:
  RX (tool):          {rx}°
  RYR (constraint):   {ryr:.3f}
  
Angle Predictions:
  Target Angle:       {final_angle:.2f}°
  Overbend Angle:     {overbend_angle:.2f}°
  Springback Factor:  {Ks:.4f}
  
Bending Strategy:
  Start Position:     Neutral (0°, 0)
  End Position:       ({rx}°, {ryr:.3f})
  Motion Type:        Linear interpolation
  Springback Comp:    {(overbend_angle - final_angle):.2f}° extra
  
Robot Controller Format:
  Columns: {', '.join(toolpath.columns)}
  Ready for CNC import
  
{'='*60}
"""
        return summary


def create_sample_toolpath(rx=-20, ryr=1.0, overbend=52, final=49, Ks=0.94):
    """
    Create a sample toolpath for demonstration
    """
    generator = ToolpathGenerator(angular_step=1.0)
    
    toolpath = generator.generate_toolpath(
        rx_start=0,
        rx_end=rx,
        ryr_start=0,
        ryr_end=ryr,
        overbend_angle=overbend,
        final_angle=final,
        n_steps=20
    )
    
    summary = generator.generate_summary(toolpath, rx, ryr, final, overbend, Ks)
    
    return toolpath, summary


if __name__ == "__main__":
    # Demo
    print("TOOLPATH GENERATOR DEMO")
    print("="*60)
    
    toolpath, summary = create_sample_toolpath()
    
    print("\n📋 Toolpath Preview (first 5 steps):")
    print(toolpath.head())
    
    print(summary)
    
    # Export
    generator = ToolpathGenerator()
    generator.export_toolpath(toolpath, "demo_toolpath.csv")
