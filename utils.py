from tqdm.notebook import tqdm
import plotly.graph_objects as go
import plotly.express as px
import gc
import pandas as pd

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
from kneed import KneeLocator


def time_series_cluster(df_TS, params):

    # Scale data by Min Max scaler
    df_TS_transform = TimeSeriesScalerMinMax(value_range=(0.0, 1.0)).fit_transform(df_TS)

    # Find optimal number of clusters
    n_clusters = [int(i) for i in np.linspace(2,20, 10)]
    inertia = np.array([])

    for n in tqdm(n_clusters):
        # Training
        model = TimeSeriesKMeans(n_clusters=n, **cluster_params).fit(df_TS_transform)
        inertia = np.append(inertia, model.inertia_)
        del model
        gc.collect()

    # Visualize elbow curve
    kneedle = KneeLocator(n_clusters, inertia, S=3, curve='convex', direction='decreasing')
    k_best = kneedle.knee

    if k_best is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=n_clusters, y=inertia, mode='lines+markers'))
        fig.add_shape(
                    type='line',
                    x0=k_best,
                    y0=0.9,
                    x1=k_best,
                    y1=max(inertia) * 1.00001,
                    line=dict(width=3, dash='dot')
                )
        fig.show()
    else:
        print("No elbow! Try to reduce S in knee or lengthen the range of clusters !")
        k_best = 4

    return k_best
