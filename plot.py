import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_benchmark_data():
    csv_path = "assets/results.csv"
    df = pd.read_csv(csv_path)

    frameworks = [tool for tool in df['Tool']]

    operation_columns = [col for col in df.columns if col != 'Tool']

    data = {}
    for operation in operation_columns:
        data[operation] = [row / 10000 * 1000000000 for row in df[operation]]
    
    return frameworks, data


def create_horizontal_bar_plot(operations, frameworks, data, title, filename, color_palette=None):
    if color_palette is None:
        color_palette = plt.colormaps['Set2'](np.linspace(0, 1, len(frameworks)))

    _, ax = plt.subplots(figsize=(12, 6))

    y_positions = np.arange(len(operations))
    bar_height = 0.15

    for i, framework in enumerate(frameworks):
        values = [data[op][i] for op in operations]
        bars = ax.barh(y_positions + i * bar_height, values, bar_height, 
                      label=framework, color=color_palette[i])

        for bar in bars:
            width = bar.get_width()
            if width > 0:
                ax.text(width + max(values) * 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{width:.1f}', ha='left', va='center', fontsize=8)

    ax.set_yticks(y_positions + (len(frameworks) - 1) * bar_height / 2)
    ax.set_yticklabels(['\n'.join(op.split('-')) for op in operations], fontsize=11, fontweight='bold')

    ax.set_xlabel('Time (nanoseconds per operation)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='x')

    ax.set_xscale('log')

    plt.tight_layout()

    output_path = f"assets/{filename}"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Saved plot: {output_path}")


def main():
    frameworks, data = parse_benchmark_data()
    
    colors = plt.colormaps['Set2'](np.linspace(0, 1, len(frameworks)))
    
    memory_ops = ['Upload', 'Download', 'Copy']
    create_horizontal_bar_plot(
        memory_ops, frameworks, data,
        'Memory Operations Performance Comparison',
        'memory-operations.png',
        colors
    )
    
    point_ops = ['Inversion', 'Grayscale', 'Threshold']
    create_horizontal_bar_plot(
        point_ops, frameworks, data,
        'Point Operations Performance Comparison', 
        'point-operations.png',
        colors
    )
    
    erosion_ops = ['Erosion-3x3-Cross', 'Erosion-3x3-Square', 'Erosion-1x3+3x1-Square']
    create_horizontal_bar_plot(
        erosion_ops, frameworks, data,
        'Erosion Operations Performance Comparison',
        'erosion-operations.png', 
        colors
    )
    
    conv3x3_ops = ['Convolution-3x3', 'Convolution-1x3+3x1', 'Gaussian-Blur-3x3']
    create_horizontal_bar_plot(
        conv3x3_ops, frameworks, data,
        '3x3 Convolution Operations Performance Comparison',
        'convolution-3x3-operations.png',
        colors
    )
    
    conv5x5_ops = ['Convolution-5x5', 'Convolution-1x5+5x1']
    create_horizontal_bar_plot(
        conv5x5_ops, frameworks, data,
        '5x5 Convolution Operations Performance Comparison',
        'convolution-5x5-operations.png',
        colors
    )


if __name__ == "__main__":
    main()
