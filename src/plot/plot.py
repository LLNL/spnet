# Global imports
import os
import collections
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.signal import savgol_filter

# Local imports
import load_tb

# Set display precision to one element past decimal
pd.set_option('display.precision', 1)


# Helper function to display comparison of rows in DF
def compare_rows(table_df, row_dict):
    # Highlight max value in each column for subset of rows
    def highlight_max_subset(col, subset_rows, color='lightblue'):
        max_value = col.loc[subset_rows].max()
        return ['background-color: {}'.format(color) if v == max_value else '' for v in col]

    row_keys = list(row_dict.keys())
    row_vals = list(row_dict.values())
    
    row1 = table_df.T[row_keys[0]]
    row2 = table_df.T[row_keys[1]]
    row_diff = row2 - row1
    for i, x in row_diff.items():
        if x > 0:
            row_diff[i] = '+{:.1f}'.format(x)
        else:
            row_diff[i] = '{:.1f}'.format(x) 

    diff_index = '{} vs. {}'.format(row_vals[1], row_vals[0])
    diff_df = pd.DataFrame({diff_index: row_diff}).T
    new_df = pd.concat([table_df, diff_df])
    styled = new_df.style.apply(highlight_max_subset, subset_rows=table_df.index).format(precision=1)

    # Highlight first char of string in cell: green if positive, red if negative
    def highlight_first_char(cell):
        if cell.startswith('+'):
            return 'background-color: lightgreen'
        elif cell.startswith('-'):
            return 'background-color: red'
        return ''
    
    ## Apply additional styling on the existing Styler
    styled_with_highlight = styled.applymap(highlight_first_char, subset=pd.IndexSlice[diff_index, :])
    
    return styled_with_highlight

# Get table showing metrics for each trial in dict, for provided epoch number
def get_epoch_data(data_dir_dict, key_dict, epoch=30,
        prod_factor=100.0, compare_row_dict=None):
    table_dict = {k:{} for k in key_dict.values()}
    for trial_name, (root_data_dir, data_dir) in data_dir_dict.items():
        data_dir_path = os.path.join(root_data_dir, data_dir)
        # Load tensorboard data as pandas dataframe
        df = load_tb.convert_tb_data(data_dir_path)
        # Get mask for provided epoch
        if epoch == 0:
            step_val = 0
        else:
            epoch_mask = (df.name == 'epoch') & (df.value == (epoch - 1))
            # Get mask for step val corresponding to provided epoch
            step_val = df[epoch_mask].step.max()
        step_mask = df.step == step_val
        # Get needed values from provided epoch
        for df_key, key_name in key_dict.items():
            if epoch is None:
                key_mask = (df.name == df_key)
            else:
                key_mask = (df.name == df_key) & step_mask
            #value = df[key_mask].value.item()
            value = df[key_mask].value.tolist()[0]
            table_dict[key_name][trial_name] = value * prod_factor
    # Convert results to DF
    table_df = pd.DataFrame(table_dict)

    # Style results
    if compare_row_dict is None:
        ## Highlight top value per trial
        table_styler = table_df.style.highlight_max(color='lightblue', axis=0).format(precision=1)
    else:
        ## Highlight top values, and also compare two rows
        table_styler = compare_rows(table_df, compare_row_dict)

    # Show results
    display(table_styler)
    return table_df

# Get all metric data for all epochs
def get_epoch_all_data(data_dir_dict, key_dict, plot_vs='epoch', organize_by='trial', prod_factor=100.0, max_iter=None):
    if organize_by == 'trial':
        table_dict = {k:{} for k in key_dict.values()}
    elif organize_by == 'metric':
        table_dict = {k:{} for k in data_dir_dict.keys()}
        print(table_dict)
    for trial_name, (root_data_dir, data_dir) in data_dir_dict.items():
        data_dir_path = os.path.join(root_data_dir, data_dir)
        # Load tensorboard data as pandas dataframe
        df = load_tb.convert_tb_data(data_dir_path)
        if plot_vs == 'epoch':
            # Get mask for provided epoch
            epoch_mask = df.name == 'epoch'
            # Get mask for step val corresponding to provided epoch
            step_val = df[epoch_mask].step.tolist()
            epoch_val = (df[epoch_mask].value.astype(int)+1).tolist()
            epoch_dict = dict(zip(step_val, epoch_val))
        # Get needed values from provided epoch
        for df_key, key_name in key_dict.items():
            key_mask = df.name == df_key
            if max_iter is not None:
                step_mask = df[key_mask].step <= max_iter
                key_mask = key_mask & step_mask
                value = df[key_mask].value
                step_list = df[key_mask].step.tolist()
            else:
                value = df[key_mask].value
                step_list = df[key_mask].step.tolist()
            if plot_vs == 'epoch':
                x_list = [epoch_dict[step] for step in step_list]
            elif plot_vs == 'iter':
                x_list = step_list
            plot_dict = {
                    'epoch': x_list,
                    'value': (value * prod_factor).tolist(),
                }
            if organize_by == 'trial':
                table_dict[key_name][trial_name] = plot_dict
            elif organize_by == 'metric':
                table_dict[trial_name][key_name] = plot_dict
    # Return results
    return table_dict

# Plot metrics vs. epochs, compared across trials
def plot_compare_trials(data_dir_dict, key_dict, prod_factor=100.0, log_yscale=False,
        plot_vs='iter', smooth=False, smooth_val=None, max_iter=None,
        dpi=300, plot_dir=None, file_name=None, epoch_all_dict=None, plot_width=4, plot_height=3):
    
    # Get data
    if epoch_all_dict is None:
        epoch_all_dict = get_epoch_all_data(data_dir_dict, key_dict, plot_vs=plot_vs, organize_by='trial', prod_factor=prod_factor, max_iter=max_iter)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']    
    # X axis label
    if plot_vs == 'epoch':
        x_axis_label = 'Epoch'
    elif plot_vs == 'iter':
        x_axis_label = 'Iter'
    # Plot data
    fig, ax_arr = plt.subplots(ncols=len(epoch_all_dict), figsize=(plot_width*len(epoch_all_dict), plot_height), dpi=dpi)
    if len(epoch_all_dict) == 1:
        ax_arr = [ax_arr]
    subfigure_letter_list = list(string.ascii_lowercase)
    for ax_idx, (ax, (key, _trial_dict)) in enumerate(zip(ax_arr, epoch_all_dict.items())):

        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(key)
        ###
        title_str = '{}\nvs. {}'.format(key, x_axis_label)
        per_plot_title_str = r'$\bf{{{}}}$'.format('({})'.format(subfigure_letter_list[ax_idx]))
        raw_title_str = r'{} {}'.format(per_plot_title_str, title_str)
        ###
        ax.set_title(raw_title_str)
        for trial_idx, (trial_name, plot_dict) in enumerate(_trial_dict.items()):
            # Optional smoothing of data
            if smooth:
                if type(smooth_val) == dict:
                    _smooth_val = smooth_val[trial_name]
                elif type(smooth_val) == int:
                    _smooth_val = smooth_val
                else:
                    raise Exception
                plot_value = savgol_filter(plot_dict['value'], 51, 1)
                smooth_plot_value = savgol_filter(plot_dict['value'], _smooth_val, 1)
                #
                #ax.plot(plot_dict['epoch'][_smooth_val:-_smooth_val], plot_value[_smooth_val:-_smooth_val], alpha=0.5, color=colors[trial_idx], lw=1)
                ax.plot(plot_dict['epoch'][_smooth_val:-_smooth_val], smooth_plot_value[_smooth_val:-_smooth_val], label=trial_name, color=colors[trial_idx], lw=1.75)
            else:
                plot_value = plot_dict['value']
                ax.plot(plot_dict['epoch'], plot_value, label=trial_name)

        # Tick labels
        try:
            plt.gca().yaxis.set_major_formatter(ScalarFormatter()) 
            plt.gca().yaxis.set_minor_formatter(ScalarFormatter());   # <---- OK
        except AttributeError:
            pass
            print('fail')

        # Log y scale
        if log_yscale:
            ax.set_yscale('log')
        # Show plot legend
        #if ax_idx == 0:#(len(ax_arr) - 1):
        if ax_idx == (len(ax_arr) - 1):
            print('Plotting legend!')
            ax.legend()
            #ax.legend(ncols=4, loc='lower left', bbox_to_anchor=(0.0, -0.5))

        # grid lines
        ax.yaxis.grid(True, which='major', color='grey', alpha=.25)

    # Tight layout
    plt.tight_layout()

    # save fig as pdf in paper plot dir
    if file_name is not None:
        file_path = os.path.join(plot_dir, file_name)
        plt.savefig(file_path,bbox_inches='tight')
    
    # Show plots
    plt.show()

def old():
    # Plot data
    fig, ax_arr = plt.subplots(ncols=len(epoch_all_dict), figsize=(4*len(epoch_all_dict), 3))
    for ax, (key, _trial_dict) in zip(ax_arr, epoch_all_dict.items()):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(key)
        ax.set_title('{} vs. Epoch'.format(key))
        for trial_name, plot_dict in _trial_dict.items():
            ax.plot(plot_dict['epoch'], plot_dict['value'], label=trial_name, marker='o')
            # Log y scale
            if log_yscale:
                ax.set_yscale('log')
            ax.legend()
    # Tight layout
    plt.tight_layout()

    # save fig as pdf in paper plot dir
    if file_name is not None:
        file_path = os.path.join(plot_dir, file_name)
        plt.savefig(file_path)
    
    # Show plots
    plt.show()

# Plot metrics vs. epochs, compared across metrics (e.g., train loss vs. val loss)
def plot_compare_metrics(data_dir_dict, key_dict, compare_key='???', plot_vs='epoch', prod_factor=100.0, smooth=False, smooth_val=1001,
        log_yscale=False, dpi=300, plot_dir=None, file_name=None):
    # X axis label
    if plot_vs == 'epoch':
        x_axis_label = 'Epoch'
    elif plot_vs == 'iter':
        x_axis_label = 'Iter'
    # Get data
    epoch_all_dict = get_epoch_all_data(data_dir_dict, key_dict, plot_vs=plot_vs, organize_by='metric', prod_factor=prod_factor)
    # Plot data
    fig, ax_arr = plt.subplots(ncols=len(epoch_all_dict), figsize=(4*len(epoch_all_dict), 3), dpi=dpi)
    if len(epoch_all_dict) == 1:
        ax_arr = [ax_arr]
    for ax, (key, _trial_dict) in zip(ax_arr, epoch_all_dict.items()):
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(compare_key)
        ax.set_title('{}\n{} vs. {}'.format(key, compare_key, x_axis_label))
        for trial_name, plot_dict in _trial_dict.items():
            # Optional smoothing of data
            if smooth:
                if type(smooth_val) == dict:
                    _smooth_val = smooth_val[trial_name]
                elif type(smooth_val) == int:
                    _smooth_val = smooth_val
                else:
                    raise Exception
                plot_value = savgol_filter(plot_dict['value'], _smooth_val, 1)
            else:
                plot_value = plot_dict['value']
            # Plot metric vs. time
            if smooth:
                ax.plot(plot_dict['epoch'][smooth_val:-smooth_val], plot_value[smooth_val:-smooth_val], label=trial_name)
            else:
                ax.plot(plot_dict['epoch'], plot_value, label=trial_name)
            # Log y scale
            if log_yscale:
                ax.set_yscale('log')
            # Show plot legend
            ax.legend()
    # Tight layout
    plt.tight_layout()

    # save fig as pdf in paper plot dir
    if file_name is not None:
        file_path = os.path.join(plot_dir, file_name)
        plt.savefig(file_path)

    # Show plots
    plt.show()

# Bar chart
def plot_bar_chart(table_df, metric_key_list, baseline_key, ybound_list=[], show_values=False, dpi=300, plot_dir=None, file_name=None, plot_width=4, plot_height=3,
        start_color_idx=0, title_fontsize=18, axis_fontsize=16, tick_fontsize=14, ax_lw=1.5, title_vs='QC vs. OC'):
    # Get colors
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    color_dict = dict(zip(['Baseline', 'PT', 'PT-Rep'], colors))

    # Get plot labels
    subfigure_letter_list = list(string.ascii_lowercase)
    
    # Setup plot
    fig, ax_arr = plt.subplots(ncols=len(metric_key_list), nrows=1, layout='constrained', figsize=(plot_width*len(metric_key_list), plot_height), dpi=dpi)
    if type(ax_arr) != np.ndarray:
        ax_arr = [ax_arr]
    for metric_idx, metric_key in enumerate(metric_key_list):
        ax = ax_arr[metric_idx]
        #
        model_dict = collections.defaultdict(dict)
        mode_dict = collections.defaultdict(dict)
        for model_key, val in table_df[metric_key].items():
            if model_key != baseline_key:
                model_name, train_mode = model_key.split(' ')
                model_dict[model_name][train_mode] = val
                mode_dict[train_mode][model_name] = val
        model_list = list(model_dict.keys())
    
        # plot
        x = np.arange(len(model_list))  # the label locations
        width = 0.3  # the width of the bars
        multiplier = 0
        
        
        for mode_idx, (attribute, measurement) in enumerate(mode_dict.items()):
            #print(attribute, measurement)
            values = list(measurement.values())
            if len(values) == 1:
                values = [0,0,0,0]+values
            offset = width * multiplier
            #print(attribute)
            rects = ax.bar(x + offset, values, width, label=attribute, alpha=0.8, edgecolor='black', color=color_dict[attribute], lw=ax_lw)
            if show_values:
                ax.bar_label(rects, fmt='{:.1f}', padding=3, fontsize='small')
            multiplier += 1

            # Monkey patch to fix legend issue
            if mode_idx == 1:
                rects[2].set_facecolor(color_dict['PT-Rep'])
                rects[3].set_facecolor(color_dict['PT-Rep'])
    
        # Plot baseline as dotted line
        if baseline_key is not None:
            by = table_df[metric_key][baseline_key]
            #bx = (by - ymin) / (ymax - ymin)
            ax.axhline(table_df[metric_key][baseline_key], linestyle='--', color='black', label='Baseline', lw=ax_lw)
            #ax.text(x=-0.65, y=by, s='{:.1f}'.format(table_df[metric_key][baseline_key]), fontsize='x-small', fontweight='bold')

        ### Prep title
        title_str = f'{metric_key} for {title_vs}'
        per_plot_title_str = r'$\bf{{{}}}$'.format('({})'.format(subfigure_letter_list[metric_idx]))
        raw_title_str = r'{} {}'.format(per_plot_title_str, title_str)
        ###
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(metric_key, fontsize=axis_fontsize)
        ax.set_title(raw_title_str, fontsize=title_fontsize)
        ax.set_xticks(x + width, model_list, rotation=45, fontsize=tick_fontsize)
        legend = ax.legend(title='Pre-Train Mode', loc='upper left', ncols=3, fontsize=10)
        legend.get_title().set_fontsize(10)
        legend.get_frame().set_linewidth(1)
        ymin, ymax = ybound_list[metric_idx]
        ax.set_ylim(ymin, ymax)
        
        # grid lines
        ax.yaxis.grid(True, which='major', color='grey', alpha=.25, lw=1)

        # line thickness
        ax.xaxis.set_tick_params(width=ax_lw)
        ax.yaxis.set_tick_params(width=ax_lw)
        [spine.set_linewidth(ax_lw) for spine in ax.spines.values()]

    # Tight layout    
    plt.tight_layout()

    # save fig as pdf in paper plot dir
    if file_name is not None:
        file_path = os.path.join(plot_dir, file_name)
        plt.savefig(file_path, bbox_inches='tight')

    #
    plt.show()

# Line plot
def plot_line_plot(table_df, metric_key_list, baseline_key=None, ybound_list=[], dpi=300, plot_dir=None, file_name=None, title=None,
        per_plot_title=None, ax_arr=None, fig=None, show=False):
    if ax_arr is None:
        fig, ax_arr = plt.subplots(ncols=len(metric_key_list), nrows=1, layout='constrained', figsize=(3*len(metric_key_list), 3), dpi=dpi)

    if baseline_key is not None:
        handle_list = []
    for metric_idx, metric_key in enumerate(metric_key_list):
        ax = ax_arr[metric_idx]
        #
        mode_x_dict = collections.defaultdict(list)
        mode_y_dict = collections.defaultdict(list)

        for model_key, val in table_df[metric_key].items():
            if model_key != baseline_key:
                p_str, train_mode = model_key.split(' ')
                p_val = int(p_str[1:])
                mode_y_dict[train_mode].append(val)
                mode_x_dict[train_mode].append(p_val)

        #
        for train_mode in mode_x_dict:
            x = mode_x_dict[train_mode]
            y = mode_y_dict[train_mode]
            handle = ax.plot(x, y, 'o-', label=train_mode)
            if (metric_idx == 0) and (baseline_key is not None):
                handle_list.extend(handle)
            
    
        # Plot baseline as dotted line
        if baseline_key is not None:
            by = table_df[metric_key][baseline_key]
            baseline_handle = ax.axhline(table_df[metric_key][baseline_key], linestyle='--', color='black', label='Baseline')
            #ax.text(x=-0.65, y=by, s='{:.1f}'.format(table_df[metric_key][baseline_key]), fontsize='x-small', fontweight='bold')
            if metric_idx == 0:
                handle_list.append(baseline_handle)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ymin, ymax = ybound_list[metric_idx]
        ax.set_ylabel(metric_key)
        ax.set_xlabel('% Annotations Used')
        if title is None:
            ax.set_title('QC vs. OC')
        else:
            if per_plot_title is None:
                ax.set_title(f'{title}\n{metric_key} for QC vs. OC')
            else:
                title_str = f'{title}\n{metric_key} for QC vs. OC'
                per_plot_title_str = r'$\bf{{{}}}$'.format(per_plot_title[metric_idx])
                print(per_plot_title_str)
                raw_title_str = r'{} {}'.format(per_plot_title_str, title_str)
                ax.set_title(raw_title_str)
        #ax.set_xticks(x + width, model_list)
        if False:
            ax.set_yticks(np.arange(ymin, ymax+1))
        #legend = ax.legend(title='Pre-Train Mode', loc='upper left', ncols=2, handles=handle_list)
        ax.set_ylim(ymin, ymax)
        
        # grid lines
        ax.yaxis.grid(True, which='major', color='grey', alpha=.25)
    
    #
    if (baseline_key is not None):
        fig.legend(title='Pre-Train Mode', handles=handle_list, loc='upper right', ncol=1, bbox_to_anchor=(1.17, 0.9))
    
    #
    plt.tight_layout()

    # save fig as pdf in paper plot dir
    if file_name is not None:
        file_path = os.path.join(plot_dir, file_name)
        plt.savefig(file_path,bbox_inches='tight')

    # show plot
    if show:
        plt.show()

# Table plotting
def bold_extreme_values(data, format_string="%.1f", max_=True):
    if max_:
        extrema = data != data.max()
    else:
        extrema = data != data.min()
    bolded = data.apply(lambda x : "\\textbf{%s}" % format_string % x)
    formatted = data.apply(lambda x : format_string % x)
    return formatted.where(extrema, bolded) 

def table_bold_max(table_df, show_latex=False):
    # bold
    l = lambda data : bold_extreme_values(data, max_=True) 
    latex_df = table_df.apply(l, axis=0)
    
    # show latex table
    if show_latex:
        for key in ['\u0394 mAP', '\u0394 top1']:
            if key in latex_df:
                for i, rv in enumerate(latex_df[key]):
                    bold = False
                    if 'textbf' in rv:
                        v = rv[-4:-1]
                        bold = True
                    else:
                        v = rv 
                    if float(v) > 0:
                        b = '+{}'.format(v)
                        if bold:
                            b = '\textbf{{{}}}'.format(b)
                        latex_df[key][i] = b
        print(latex_df.to_latex(escape=False))

    # display
    #table_df = table_df.style.highlight_max(color = 'lightgreen', axis = 0)
    #display(table_df)
