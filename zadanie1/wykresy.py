import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def make_plot_for_all(df, title, xLabel, yLabel, width, is_log):
    print(df)
    algorithms = ['bfs', 'dfs', 'astar']
    depths = df.index.tolist()
    x = np.arange(len(depths))  # pozycje grup

    base_width = 8
    extra_width_per_alg = 0.5
    fig_width = base_width + len(algorithms) * extra_width_per_alg
    plt.figure(figsize=(fig_width, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # kolory: BFS, DFS, A*

    for i, algo in enumerate(algorithms):
        plt.bar(x + i * width,
                df[algo],
                width=width,
                label=algo.upper(),
                color=colors[i],)
    plt.title(title, fontsize=23)
    plt.xlabel(xLabel,size=20)
    plt.ylabel(yLabel,size=20)
    if is_log:
        plt.yscale('log')
    plt.xticks(x + width, depths, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='center left',
               bbox_to_anchor=(1.0, 0.5),
               ncol=1,
               fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def make_data_frames(name):
    bfs_name = {depth: bfs_data[bfs_data['depth'] == depth][name].mean() for depth in range(1, 8)}
    dfs_name = {depth: dfs_data[dfs_data['depth'] == depth][name].mean() for depth in range(1, 8)}
    astar_name = {depth: astar_data[astar_data['depth'] == depth][name].mean() for depth in range(1, 8)}
    name_data = pd.DataFrame({
        'bfs': list(bfs_name.values()),
        'dfs': list(dfs_name.values()),
        'astar': list(astar_name.values())
    }, index=range(1, 8))

    params = ['drlu', 'drul', 'ludr', 'lurd', 'rdlu', 'rdul', 'uldr', 'ulrd']
    a_params = ['hamm', 'manh']
    bfs_params = []
    dfs_params = []
    astar_params = []

    for depth in range(1, 8):
        for param in params:
            mean_val_bfs = bfs_data[(bfs_data['depth'] == depth) & (bfs_data['param'] == param)][name].mean()
            bfs_params.append((depth, param, mean_val_bfs))
            mean_val_dfs = dfs_data[(dfs_data['depth'] == depth) & (dfs_data['param'] == param)][name].mean()
            dfs_params.append((depth, param, mean_val_dfs))
        for param in a_params:
            mean_val_astar = astar_data[(astar_data['depth'] == depth) & (astar_data['param'] == param)][name].mean()
            astar_params.append((depth, param, mean_val_astar))

    bfs_df = pd.DataFrame(bfs_params, columns=['depth', 'param', 'mean_val'])
    dfs_df = pd.DataFrame(dfs_params, columns=['depth', 'param', 'mean_val'])
    astar_df = pd.DataFrame(astar_params, columns=['depth', 'param', 'mean_val'])
    return bfs_df, dfs_df, astar_df,name_data

def make_distinct_plot(params, df, title, xLabel, yLabel, width, is_log):
    print(df)
    global values
    depths = [1, 2, 3, 4, 5, 6, 7]
    x = np.arange(len(depths))
    plt.figure(figsize=(10, 6))
    for i, param in enumerate(params):
        values = []
        for depth in depths:
            filtered = df[(df['depth'] == depth) & (df['param'] == param)]
            mean_val = filtered['mean_val'].values[0] if not filtered.empty else 0
            values.append(mean_val)
        # przesunięcie względem pozycji depth
        plt.bar(x + i * width, values, width=width, label=param.upper())

    center_offset = (len(params) - 1) / 2 * width
    plt.title(title, fontsize=23)
    plt.xlabel(xLabel, size=20)
    plt.ylabel(yLabel, size=20)
    if is_log:
        plt.yscale('log')
    plt.xticks(x + center_offset, depths, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='center left',
               bbox_to_anchor=(1.0, 0.5),
               ncol=1,
               fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

data=pd.read_csv('ekstrakcja_danych.txt',
                sep=r'\s+',            # użycie spacji/tabów jako separatorów
                header = None,
                encoding='UTF-16 LE',
                names = ['depth', 'example', 'algorithm', 'param', 'solution_length','open_states_num','processed_states_num','max_recursion_depth','processing_time'])

data = data[data['solution_length'] != -1]
dfs_data = data[data['algorithm'] == 'dfs']
bfs_data = data[data['algorithm'] == 'bfs']
astar_data = data[data['algorithm'] == 'astr']
params = ['drlu', 'drul', 'ludr', 'lurd', 'rdlu', 'rdul', 'uldr', 'ulrd']
astar_params = ['hamm', 'manh']

# długość znalezionego rozwiązania
bfs_sol_len_df, dfs_sol_len_df, astar_sol_len_df,sol_len_data=make_data_frames('solution_length')
    #ogółem
make_plot_for_all(sol_len_data, 'Średnia długość rozwiązania\n dla różnych strategii', 'Głębokość',
                  'Średnia długość rozwiązania', 0.25, False)
    # bfs
make_distinct_plot(params, bfs_sol_len_df, 'Średnia długość rozwiązania\n dla algorytmu BFS', 'Głębokość',
                   'Średnia długość rozwiązania', 0.1, False)

    # dfs
make_distinct_plot(params, dfs_sol_len_df, 'Średnia długość rozwiązania\n dla algorytmu DFS', 'Głębokość',
                   'Średnia długość rozwiązania', 0.1, False)

    # astr
make_distinct_plot(astar_params, astar_sol_len_df, 'Średnia długość rozwiązania\n dla algorytmu A*', 'Głębokość',
                   'Średnia długość rozwiązania', 0.4, False)

# liczba stanów odwiedzonych
bfs_open_states_df, dfs_open_states_df, astar_open_states_df,open_states_data=make_data_frames('open_states_num')

    # ogółem
make_plot_for_all(open_states_data, 'Średnia liczba stanów odwiedzonych\n dla różnych strategii', 'Głębokość',
                  'Średnia liczba stanów odwiedzonych', 0.25, True)

    # bfs
make_distinct_plot(params, bfs_open_states_df, 'Średnia liczba stanów odwiedzonych\n dla algorytmu BFS', 'Głębokość',
                   'Średnia liczba stanów odwiedzonych', 0.1, True)

    # dfs
make_distinct_plot(params, dfs_open_states_df, 'Średnia liczba stanów odwiedzonych\n dla algorytmu DFS', 'Głębokość',
                   'Średnia liczba stanów odwiedzonych', 0.1, True)

    # astr
make_distinct_plot(astar_params, astar_open_states_df, 'Średnia liczba stanów odwiedzonych\n dla algorytmu A*',
                   'Głębokość', 'Średnia liczba stanów odwiedzonych', 0.4, False)

# liczba stanów przetworzonych
bfs_proc_states_df, dfs_proc_states_df, astar_proc_states_df,proc_states_data=make_data_frames('processed_states_num')

    # ogółem
make_plot_for_all(proc_states_data, 'Średnia liczba stanów przetworzonych\n dla różnych strategii', 'Głębokość',
                  'Średnia liczba stanów przetworzonych', 0.25, True)

    # bfs
make_distinct_plot(params, bfs_proc_states_df, 'Średnia liczba stanów przetworzonych\n dla algorytmu BFS', 'Głębokość',
                   'Średnia liczba stanów przetworzonych', 0.1, True)

    # dfs
make_distinct_plot(params, dfs_proc_states_df, 'Średnia liczba stanów przetworzonych\n dla algorytmu DFS', 'Głębokość',
                   'Średnia liczba stanów przetworzonych', 0.1, True)

    # astr
make_distinct_plot(astar_params, astar_proc_states_df, 'Średnia liczba stanów przetworzonych\n dla algorytmu A*',
                   'Głębokość', 'Średnia liczba stanów przetworzonych', 0.4, False)


# maksymalna osiągnięta głębokość rekursji
bfs_max_dep_df, dfs_max_dep_df, astar_max_dep_df,max_dep_data=make_data_frames('max_recursion_depth')

    # ogółem
make_plot_for_all(max_dep_data, 'Średnia liczba maksymalnej głębokości\n rekursji dla różnych strategii', 'Głębokość',
                  'Średnia liczba maksymalnej\n głębokości rekursji', 0.25, False)

    # bfs
make_distinct_plot(params, bfs_max_dep_df, 'Średnia liczba maksymalnej głębokości\n rekursji dla algorytmu BFS',
                   'Głębokość', 'Średnia liczba maksymalnej\n głębokości rekursji', 0.1, False)

    # dfs
make_distinct_plot(params, dfs_max_dep_df, 'Średnia liczba maksymalnej głębokości\n rekursji dla algorytmu DFS',
                   'Głębokość', 'Średnia liczba maksymalnej\n głębokości rekursji', 0.1, False)

    # astr
make_distinct_plot(astar_params, astar_max_dep_df, 'Średnia liczba maksymalnej głębokości\n rekursji dla algorytmu A*',
                   'Głębokość', 'Średnia liczba maksymalnej\n głębokości rekursji', 0.4, False)

# czas trwania procesu obliczeniowego
bfs_time_df, dfs_time_df, astar_time_df,time_data=make_data_frames('processing_time')

    # ogółem
make_plot_for_all(time_data, 'Średnia długość czasu trwania procesu\n obliczeniowego dla różnych strategii',
                  'Głębokość', 'Średnia długość czasu trwania procesu\n obliczeniowego', 0.25, True)

    # bfs
make_distinct_plot(params, bfs_time_df,
                   'Średnia długość czasu trwania procesu\n obliczeniowego dla strategii algorytmu BFS', 'Głębokość',
                   'Średnia długość czasu trwania procesu\n obliczeniowego', 0.1, True)

    # dfs
make_distinct_plot(params, dfs_time_df,
                   'Średnia długość czasu trwania procesu\n obliczeniowego dla strategii algorytmu DFS', 'Głębokość',
                   'Średnia długość czasu trwania procesu\n obliczeniowego', 0.1, True)

    # astr
make_distinct_plot(astar_params, astar_time_df,
                   'Średnia długość czasu trwania procesu\n obliczeniowego dla strategii algorytmu A*', 'Głębokość',
                   'Średnia długość czasu trwania procesu\n obliczeniowego', 0.4, True)
