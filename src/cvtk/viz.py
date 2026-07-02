import numpy as np
import pandas as pd
import plotly.subplots
import plotly.graph_objects as go
import plotly.express as px
import sklearn.metrics


def plot(log_fpath, y, output=None, title='Plot', mode='lines', width=600, height=800, 
         scale=1.0, rows=None, cols=None, x='epoch'):
    """Generic plotting function to plot specified columns from a tab-separated file.

    Plot specified columns from a tab-separated log file.
    Supports grouping multiple columns in the same subplot using nested lists.

    Args:
        log_fpath (str): Path to tab-separated log file.
        y (list): List of column names to plot or nested lists for grouped subplots.
            Examples:
                - ['loss', 'acc']: plots loss and acc in separate subplots
                - [['train_loss', 'valid_loss'], ['train_acc', 'valid_acc']]:
                  plots train_loss and valid_loss in subplot 1,
                  train_acc and valid_acc in subplot 2
        output (str|None): Output file path to save the plot. If None, plot is displayed.
        title (str): Title of the plot. Default is 'Plot'.
        mode (str): Plot mode ('lines', 'markers', etc.). Default is 'lines'.
        width (int): Width of the plot in pixels. Default is 600.
        height (int): Height of the plot in pixels. Default is 800.
        scale (float): Scale factor for the plot. Default is 1.0.
        rows (int|None): Number of rows in subplot grid. If None, auto-generated for near square layout.
        cols (int|None): Number of columns in subplot grid. If None, auto-generated for near square layout.
        x (str): Column name for x-axis. Default is 'epoch'.

    Returns:
        plotly.graph_objects.Figure: The plotly figure object.

    Examples:
        >>> from cvtk.utils import plot
        >>> plot('train.log', y=['loss', 'acc'], output='plot.png')
        >>> plot('train.log', y=[['train_loss', 'valid_loss'], ['train_acc', 'valid_acc']])
    """
    log_data = pd.read_csv(log_fpath, sep='\t', header=0, comment='#')
    
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


def plot_cm(test_outputs, output=None, title='Confusion Matrix', xlab='Predicted Label', ylab='True Label', colorscale='YlOrRd', width=600, height=600, scale=1.0):
    """Plot a confusion matrix from test outputs

    Plot a confusion matrix from test outputs.
    The test outputs are saved in a tab-separated file,
    where the first column is the path to the image, the second column is the true label,
    and the following columns are the predicted probabilities for each class.
    The example of the test outputs is as follows:

    ::

        image  label   leaf     flower   root
        1.JPG  leaf    0.54791  0.20376  0.24833
        2.JPG  root    0.06158  0.02184  0.91658
        3.JPG  leaf    0.70320  0.04808  0.24872
        4.JPG  flower  0.04723  0.90061  0.05216
        5.JPG  flower  0.30027  0.63067  0.06906
        6.JPG  leaf    0.52753  0.43249  0.03998
        7.JPG  root    0.21375  0.14829  0.63796
    

    Args:
        test_outputs (str): A path to a tab-separated file containing test outputs.
        output (str): A file path to save the output images. If not provided, the plot is shown on display.
        width (int): A width of the output image.
        height (int): A height of the output image.
        scale (float): The scale of the output image, which is used to adjust the resolution.
    """
    # data preparation
    test_outputs = pd.read_csv(test_outputs, sep='\t', header=0, comment='#')
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
        import os
        fig.write_image(output, width=width, height=height, scale=scale)
        cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
        with open(os.path.splitext(output)[0] + '.txt', 'w') as oufh:
            oufh.write('# Confusion Matrix\n')
            oufh.write('#\tprediction\n')
            cm.to_csv(oufh, sep='\t', header=True, index=True)
    else:
        fig.show()

    return fig

