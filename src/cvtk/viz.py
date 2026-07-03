import os
import numpy as np
import pandas as pd
import plotly.subplots
import plotly.graph_objects as go
import plotly.express as px
import sklearn.metrics


def plot(
    data,
    x=None,
    y=None,
    output=None,
    title=None,
    mode='lines',
    width=600,
    height=800, 
    scale=1.0,
    rows=None,
    cols=None
) -> go.Figure:
    """Plot specified columns from a tab-separated log file.
    
    Reads a tab-separated file and creates line plots using Plotly. Supports
    multiple subplots where each subplot can contain one or more y columns.
    
    Args:
        data (str): Path to tab-separated log file.
        x (str|None): Column name for x-axis. If None, defaults to 'epoch'. Default is None.
        y (str|list): Column name(s) to plot on y-axis. Can be a single column name (str)
            or a list of column names/nested lists for grouped subplots:
                - 'loss': plots single column
                - ['loss', 'acc']: plots loss and acc in separate subplots
                - [['train_loss', 'valid_loss'], ['train_acc', 'valid_acc']]:
                  plots train_loss and valid_loss in subplot 1,
                  train_acc and valid_acc in subplot 2
        output (str|None): File path to save the plot. If None, displays plot interactively.
        title (str|None): Plot title. Default is None.
        mode (str): Plotly trace mode ('lines', 'markers', 'lines+markers', etc.). Default is 'lines'.
        width (int): Plot width in pixels. Default is 600.
        height (int): Plot height in pixels. Default is 800.
        scale (float): Scale factor for saved image resolution. Default is 1.0.
        rows (int|None): Number of rows in subplot grid. If None, auto-calculated for near-square layout.
        cols (int|None): Number of columns in subplot grid. If None, auto-calculated for near-square layout.

    Returns:
        plotly.graph_objects.Figure: The plotly figure object.

    Raises:
        TypeError: If y items are not str or list/tuple.
        ValueError: If specified column names are not found in the data file.

    Examples:
        >>> from cvtk.viz import plot
        >>> plot('train.log', y=['loss', 'acc'], output='plot.png')
        >>> plot('train.log', x='step', y=[['train_loss', 'valid_loss'], ['train_acc', 'valid_acc']])
    """
    log_data = pd.read_csv(data, sep='\t', header=0, comment='#')
    
    # normalize y to nested list format
    if isinstance(y, str):
        y = [y]
    
    y_labels = []
    for item in y:
        if isinstance(item, str):
            y_labels.append([item])
        elif isinstance(item, (list, tuple)):
            y_labels.append(list(item))
        else:
            raise TypeError(f"y items must be str or list, got {type(item)}")
    
    n_subplots = len(y_labels)
    
    if rows is None and cols is None:
        cols = int(np.ceil(np.sqrt(n_subplots)))
        rows = int(np.ceil(n_subplots / cols))
    elif rows is None:
        rows = int(np.ceil(n_subplots / cols))
    elif cols is None:
        cols = int(np.ceil(n_subplots / rows))
    
    fig = plotly.subplots.make_subplots(rows=rows, cols=cols)
    
    colors = px.colors.qualitative.Plotly
    color_idx = 0
    
    for subplot_idx, y_cols in enumerate(y_labels):
        row = (subplot_idx // cols) + 1
        col = (subplot_idx % cols) + 1
        
        for y_col in y_cols:
            if y_col not in log_data.columns:
                raise ValueError(f"Column '{y_col}' not found in log file. Available columns: {list(log_data.columns)}")
            
            fig.add_trace(
                go.Scatter(
                    x=log_data[x],
                    y=log_data[y_col],
                    mode=mode,
                    name=y_col,
                    line=dict(color=colors[color_idx % len(colors)])
                ),
                row=row, col=col
            )
            color_idx += 1
    
    fig.update_layout(title_text=title, template='ggplot2')
    fig.update_xaxes(title_text=x)
    
    if output is not None:
        fig.write_image(output, width=width, height=height, scale=scale)
    else:
        fig.show()
    
    return fig


def plot_cm(
    data,
    output=None,
    title='Confusion Matrix',
    xlab='Predicted Label',
    ylab='True Label',
    colorscale='YlOrRd',
    width=600,
    height=600,
    scale=1.0
) -> go.Figure:
    """Plot a confusion matrix from classification test outputs.

    Plots a confusion matrix as a heatmap using Plotly. Also saves a text file
    containing the confusion matrix values if output path is provided.
    
    The input data should be a tab-separated file with columns:
    - Column 1: image/sample path
    - Column 2: true class label
    - Columns 3+: predicted probabilities for each class

    Example input format::

        image  label   leaf     flower   root
        1.JPG  leaf    0.54791  0.20376  0.24833
        2.JPG  root    0.06158  0.02184  0.91658
        3.JPG  leaf    0.70320  0.04808  0.24872
        4.JPG  flower  0.04723  0.90061  0.05216

    Args:
        data (str): Path to tab-separated file containing test outputs.
        output (str|None): File path to save the heatmap image. Also saves a `.txt` file
            with the confusion matrix values. If None, displays plot interactively.
        title (str): Plot title. Default is 'Confusion Matrix'.
        xlab (str): X-axis label. Default is 'Predicted Label'.
        ylab (str): Y-axis label. Default is 'True Label'.
        colorscale (str): Plotly colorscale name (e.g., 'YlOrRd', 'Blues', 'Viridis').
            Default is 'YlOrRd'.
        width (int): Image width in pixels. Default is 600.
        height (int): Image height in pixels. Default is 600.
        scale (float): Scale factor for saved image resolution. Default is 1.0.

    Returns:
        plotly.graph_objects.Figure: The plotly figure object.

    Examples:
        >>> from cvtk.viz import plot_cm
        >>> plot_cm('test_results.txt', output='confusion_matrix.png')
    """
    # data preparation
    test_outputs = pd.read_csv(data, sep='\t', header=0, comment='#')
    class_labels = test_outputs.columns[2:]
    y_true = test_outputs.iloc[:, 1].values.tolist()
    y_pred = test_outputs.iloc[:, 2:].idxmax(axis=1).values.tolist()
    
    # statistics calculation
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=test_outputs.columns[2:])

    fig = go.Figure(data=go.Heatmap(x=class_labels, y=class_labels, z=cm,
                                    colorscale=colorscale, hoverongaps=False))
    fig.update_layout(title=title, xaxis_title=xlab, yaxis_title=ylab,
                      xaxis=dict(side='bottom'), yaxis=dict(side='left'))
    fig.update_layout(template='ggplot2')

    if output is not None:
        fig.write_image(output, width=width, height=height, scale=scale)
        cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
        with open(os.path.splitext(output)[0] + '.txt', 'w') as oufh:
            oufh.write('# Confusion Matrix\n')
            oufh.write('#\tprediction\n')
            cm.to_csv(oufh, sep='\t', header=True, index=True)
    else:
        fig.show()

    return fig

