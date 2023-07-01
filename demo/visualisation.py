import scipy.stats as stats
import pandas as pd
import numpy as np

from bokeh.plotting import figure
from bokeh.layouts import gridplot
# from bokeh.sampledata.antibiotics import data as df

var_type = {
    "Pauses": ["MeanPauseDuration",
               "PauseFrequency",
               "PauseVoiceRatio"],
    "Loudness": ["equivalentSoundLevel_dBp",
                 "loudness_sma3_amean",
                 "loudness_std"],
    # "loudness_sma3_percentiles"],
    "Speechrate": ["VoicedSegmentsPerSec",
                   "MeanUnvoicedSegmentLength",
                   "SpeechRate_wpm"]
    # "MeanVoicedSegmentLengthSec"]
    }

var_color = dict(
    [("Pauses", "#FF7E67"), ("Loudness", "#A2D5F2"), ("Speechrate", "#07689F")]
    )


def generate_bokeh_plot(d, m, s):

    df = pd.Series(d).to_frame().T
    # mean = pd.Series(m).to_frame().T
    # std = pd.Series(s).to_frame().T
    df.rename(columns={
        "loudness_mean": "loudness_sma3_amean",
        "sound_level_db": "equivalentSoundLevel_dBp",
        "pause_average_duration": "MeanPauseDuration",
        "pause_frequency": "PauseFrequency",
        "pause_voice_ratio": "PauseVoiceRatio",
        "pseudo_syl_rate": "VoicedSegmentsPerSec",
        "unvoiced_average_duration": "MeanUnvoicedSegmentLength"
        }, errors="raise", inplace=True)
    df.drop(columns=['pitch_std'], inplace=True)
    width = 300
    height = 300
    plots = {}
    for k in var_type.keys():
        p_tmp = []
        for val in var_type[k]:
            mu, sigma = m[val], s[val]
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            y = stats.norm.pdf(x, mu, sigma)
            p = figure(title=val, x_axis_label='values',
                       y_axis_label='distribution')
            p.varea(x, y1=0, y2=y, alpha=.3, color=var_color[k])
            x_obs = df[val].values[0]
            p.diamond(x=x_obs, y=0, color='red', size=15)
            p.line(x=[x_obs, x_obs], y=[0, y.max()], color='red')
            p_tmp.append(p)
        plots[k] = gridplot(children=p_tmp, ncols=3,
                            height=height,
                            width=width
                            )
    return plots
