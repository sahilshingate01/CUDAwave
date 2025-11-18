import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    filepath = 'data/benchmark.csv'
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return
        
    df = pd.read_csv(filepath)
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    x = range(len(df['grid']))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], df['gpu_ms'], width, label='GPU', color='#1f77b4')
    ax1.bar([i + width/2 for i in x], df['cpu_ms'], width, label='CPU', color='#ff7f0e')
    
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['grid'])
    
    ax2 = ax1.twinx()
    ax2.plot(x, df['speedup'], 'r-o', linewidth=2, label='Speedup')
    ax2.set_ylabel('Speedup (x)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title("GPU vs CPU 2D FDTD Benchmark")
    plt.tight_layout()
    plt.savefig('data/benchmark.png', dpi=300)
    print("Saved data/benchmark.png")
    
    max_speedup = df['speedup'].max()
    max_grid = df.loc[df['speedup'].idxmax(), 'grid']
    print(f"Peak speedup: {max_speedup:.1f}x at {max_grid} grid")

if __name__ == '__main__':
    main()