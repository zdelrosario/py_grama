from .tools import *
from .core import *

from .comp_building import *
from .comp_metamodels import *
from .eval_defaults import *
from .eval_random import *
from .eval_tail import *
from .plot_auto import *
from .tran_summaries import *
from .tran_tools import *

# Integrate dfply tools
# --------------------------------------------------
from .dfply import Intention, dfdelegate, make_symbolic, var_in, convert_type
from .dfply import starts_with, ends_with, contains, matches, everything
from .dfply import num_range, one_of, columns_between, columns_from, columns_to

# group.py
from .dfply import group_by as tf_group_by
from .dfply import ungroup as tf_ungroup

# join.py
from .dfply import inner_join as tf_inner_join
from .dfply import full_join as tf_full_join
from .dfply import outer_join as tf_outer_join
from .dfply import left_join as tf_left_join
from .dfply import right_join as tf_right_join
from .dfply import semi_join as tf_semi_join
from .dfply import anti_join as tf_anti_join
from .dfply import bind_rows as tf_bind_rows
from .dfply import bind_cols as tf_bind_cols

# reshape.py
from .dfply import arrange as tf_arrange
from .dfply import rename as tf_rename
from .dfply import separate as tf_separate
from .dfply import unite as tf_unite
from .dfply import gather as tf_gather
from .dfply import spread as tf_spread

# select.py
from .dfply import select as tf_select
from .dfply import select_if as tf_select_if
from .dfply import drop as tf_drop
from .dfply import drop_if as tf_drop_if

# set_ops.py
from .dfply import union as tf_union
from .dfply import intersect as tf_intersect
from .dfply import set_diff as tf_set_diff

# subset.py
from .dfply import head as tf_head
from .dfply import tail as tf_tail
from .dfply import sample as tf_sample
from .dfply import distinct as tf_distinct
from .dfply import row_slice as tf_row_slice
from .dfply import mask as tf_filter
from .dfply import top_n as tf_top_n
from .dfply import pull as tf_pull

# summarize.py
from .dfply import summarize as tf_summarize
from .dfply import summarize_each as tf_summarize_each

# summary_functions.py
from .dfply import mean, first, last, nth, n, n_distinct, IQR, quant
from .dfply import colmin, colmax, median, var, sd

# transform.py
from .dfply import mutate as tf_mutate
from .dfply import mutate_if as tf_mutate_if
from .dfply import transmute as tf_transmute

# vector.py
from .dfply import order_series_by, desc, coalesce, case_when, if_else, na_if

# window_functions.py
from .dfply import lead, lag, between, dense_rank, min_rank
from .dfply import cumsum, cummean, cummax, cummin, cumprod, cumany, cumall
from .dfply import percent_rank, row_number

# Add functionality to dfply
from .string_helpers import *
