from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import confusion_matrix
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
import plotly


def gen_folds(random_state, generator, n_folds=5):
    folds_gene = StratifiedKFold(n_folds, random_state=random_state,
                                 shuffle=True)
    folds = folds_gene.split(np.arange(generator.y.shape[0]), generator.y)
    folds = [[list(train), list(test)] for train, test in folds]
    return folds


def test_dataset( folds, n_views, n_classes, generator,):
    dt = DecisionTreeClassifier(max_depth=3)
    n_folds = len(folds)
    confusion_mat = np.zeros((n_folds, n_views, n_classes, n_classes))
    n_sample_per_class = np.zeros((n_views, n_classes, n_folds))
    for view_index in range(n_views):
        for fold_index, [train, test] in enumerate(folds):
            dt.fit(generator.view_data[view_index][train, :], generator.y[train])
            pred = dt.predict(generator.view_data[view_index][test, :])
            confusion_mat[fold_index, view_index, :, :] = confusion_matrix(generator.y[test], pred)
            for class_index in range(n_classes):
                n_sample_per_class[view_index, class_index, fold_index] = np.where(generator.y[test]==class_index)[0].shape[0]
    confusion_mat = np.mean(confusion_mat, axis=0)
    n_sample_per_class = np.mean(n_sample_per_class, axis=2)
    confusion_output = np.zeros((n_classes, n_views))
    for class_index in range(n_classes):
        for view_index in range(n_views):
            confusion_output[class_index, view_index] = 1-confusion_mat[view_index, class_index, class_index]/n_sample_per_class[view_index, class_index]
    return confusion_output


def make_fig(conf, confusion_output, n_views, n_classes, generator):
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "View {}, Confusion : <br>In:{}<br>Out:{}".format(view_index,
                                                   np.round(conf[:, view_index], 3),
                                                   np.round(confusion_output[:, view_index], 3)) for
        view_index
        in range(n_views)],
                        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, ],
                               [{'type': 'scatter3d'},
                                {'type': 'scatter3d'}, ]])
    row = 1
    col = 1
    for view_index in range(n_views):
        for lab_index in range(n_classes):
            concerned_examples = np.where(generator.y == lab_index)[0]
            fig.add_trace(
                go.Scatter3d(
                    x=generator.view_data[view_index][concerned_examples, 0],
                    y=generator.view_data[view_index][concerned_examples, 1],
                    z=generator.view_data[view_index][concerned_examples, 2],
                    mode='markers', marker=dict(
                        size=1,  # set color to an array/list of desired values
                        color=DEFAULT_PLOTLY_COLORS[lab_index],
                        opacity=0.8
                    ), name="Class {}".format(lab_index)), row=row, col=col)
            # fig.update_layout(
            #             scene=dict(
            #             xaxis=dict(nticks=4, range=[low_range, high_range], ),
            #             yaxis=dict(nticks=4, range=[low_range, high_range], ),
            #             zaxis=dict(nticks=4, range=[low_range, high_range], ), ),)
        col += 1
        if col == 3:
            col = 1
            row += 1
            # fig.update_xaxes(range=[-class_sep-0.1*class_sep, +class_sep+margin_ratio*class_sep], row=row, col=col)
            # fig.update_yaxes(
            #     range=[-class_sep - 0.1 * class_sep, +class_sep + margin_ratio * class_sep],
            #     row=row, col=col)
            # fig.update_zaxes(
            #     range=[-class_sep - 0.1 * class_sep, +class_sep + margin_ratio * class_sep],
            #     row=row, col=col)
    plotly.offline.plot(fig, filename="center_blob.html")

