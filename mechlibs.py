import os
import re
import platform
import sys
import einops
from uuid import uuid4
import functools
from functools import partial
from pathlib import Path
import torch as t
from torch import Tensor
import numpy as np
from tqdm import tqdm, trange
import plotly.express as px
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from IPython.display import display, HTML
from rich.table import Table, Column
from rich import print as rprint
import circuitsvis as cv
import pandas as pd
from transformer_lens.hook_points import HookPoint
# sentencepiece failed to build for me on arch (btw). can be compiled from source on linux as insrtructed here:
# https://github.com/google/sentencepiece/tree/master/python
from transformer_lens import utils, HookedTransformer, ActivationCache, patching
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP
from plotly_utils import imshow, line, bar#, scatter
import plotly.graph_objects as go
import string
import random
from eindex import eindex
purple = '\x1b[38;2;255;0;255m'
blue = '\x1b[38;2;0;0;255m'
brown = '\x1b[38;2;128;128;0m'
cyan = '\x1b[38;2;0;255;255m'
lime = '\x1b[38;2;0;255;0m'
yellow = '\x1b[38;2;255;255;0m'
red = '\x1b[38;2;255;0;0m'
pink = '\x1b[38;2;255;51;204m'
orange = '\x1b[38;2;255;51;0m'
green = '\x1b[38;2;0;0;128m'
gray = '\x1b[38;2;127;127;127m'
magenta = '\x1b[38;2;128;0;128m'
white = '\x1b[38;2;255;255;255m'
bold = '\033[1m'
underline = '\033[4m'
endc = '\033[0m'

mechinterp_dir = os.path.dirname(os.path.realpath(__file__))
if mechinterp_dir not in sys.path: sys.path.append(mechinterp_dir)

def all_attn_patterns(cache):
    nlayers = max([int(e[7]) for e in cache.keys() if 'blocks' in e])
    return t.cat([cache['pattern', layer] for layer in range(nlayers+1)], dim=1)

def show_attn_heads(cache, dataset_idx, dataset=None):
    patterns = all_attn_patterns(cache=cache)[dataset_idx]
    n_layers = max([int(e[7]) for e in cache.keys() if 'blocks' in e])
    n_heads = cache['k', 0].shape[2]
    return cv.attention.attention_heads(
        patterns,
        attention_head_names=[f"head{i}.{j}" for i in range(n_layers+1) for j in range(n_heads+1)],
        tokens=dataset.str_toks[dataset_idx] if dataset is not None else [str(i) for i in range(patterns.shape[0])],
    )


def scatter(x, y, title="", xaxis="", yaxis="", colorbar_title="", **kwargs):
    fig = px.scatter(x=utils.to_numpy(x.flatten()), y=utils.to_numpy(y.flatten()), title=title, labels={"color": colorbar_title}, **kwargs)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    if "xaxis_range" in kwargs:
        fig.update_xaxes(range=kwargs["xaxis_range"])
    if "yaxis_range" in kwargs:
        fig.update_yaxes(range=kwargs["yaxis_range"])
    fig.show()


def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, hover=None, **kwargs):
    # Helper function to plot multiple lines
    if isinstance(lines_list, t.Tensor):
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    fig = go.Figure(layout={'title':title.replace("\n", "<br>")})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, lin in enumerate(lines_list):
        if isinstance(lin, t.Tensor):
            lin = utils.to_numpy(lin)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=lin, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()

def line_marker(x, **kwargs):
    lines([x], mode='lines+markers', **kwargs)

def animate_lines(lines_list, snapshot_index = None, snapshot='snapshot', hover=None, xaxis='x', yaxis='y', title='', **kwargs):
    if isinstance(lines_list, list):
        lines_list = t.stack(lines_list, axis=0)
    lines_list = utils.to_numpy(lines_list)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    rows=[]
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[1]):
            rows.append([lines_list[i][j], snapshot_index[i], j])
    df = pd.DataFrame(rows, columns=[yaxis, snapshot, xaxis])
    px.line(df, x=xaxis, y=yaxis, title=title, animation_frame=snapshot, range_y=[lines_list.min(), lines_list.max()], hover_name=hover,**kwargs).show()

def animate_multi_lines(lines_list, y_index=None, snapshot_index = None, snapshot='snapshot', hover=None, swap_y_animate=False, **kwargs):
    # Can plot an animation of lines with multiple lines on the plot.
    if isinstance(lines_list, list):
        lines_list = t.stack(lines_list, axis=0)
    lines_list = utils.to_numpy(lines_list)
    lines_list = lines_list.transpose(2, 0, 1)
    if swap_y_animate:
        lines_list = lines_list.transpose(1, 0, 2)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if y_index is None:
        y_index = [str(i) for i in range(lines_list.shape[1])]
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    # print(lines_list.shape)
    rows=[]
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[2]):
            rows.append(list(lines_list[i, :, j])+[snapshot_index[i], j])
    df = pd.DataFrame(rows, columns=y_index+[snapshot, 'x'])
    px.line(df, x='x', y=y_index, animation_frame=snapshot, range_y=[lines_list.min(), lines_list.max()], hover_name=hover, **kwargs).show()


def animate_scatter(lines_list, snapshot_index = None, snapshot='snapshot', hover=None, yaxis='y', xaxis='x', color=None, color_name = 'color', **kwargs):
    # Can plot an animated scatter plot
    # lines_list has shape snapshot x 2 x line
    if isinstance(lines_list, list):
        lines_list = t.stack(lines_list, axis=0)
    lines_list = utils.to_numpy(lines_list)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    if color is None:
        color = np.ones(lines_list.shape[-1])
    if isinstance(color, t.Tensor):
        color = utils.to_numpy(color)
    if len(color.shape)==1:
        color = einops.repeat(color, 'x -> snapshot x', snapshot=lines_list.shape[0])
    # print(lines_list.shape)
    rows=[]
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[2]):
            rows.append([lines_list[i, 0, j].item(), lines_list[i, 1, j].item(), snapshot_index[i], color[i, j]])
    # print([lines_list[:, 0].min(), lines_list[:, 0].max()])
    # print([lines_list[:, 1].min(), lines_list[:, 1].max()])
    df = pd.DataFrame(rows, columns=[xaxis, yaxis, snapshot, color_name])
    px.scatter(df, x=xaxis, y=yaxis, animation_frame=snapshot, range_x=[lines_list[:, 0].min(), lines_list[:, 0].max()], range_y=[lines_list[:, 1].min(), lines_list[:, 1].max()], hover_name=hover, color=color_name, **kwargs).show()
