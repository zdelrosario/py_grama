## Simple model pipeline example
import grama as gr
import pandas as pd

model = gr.core.model_()
df    = pd.DataFrame(data = {"x" : [0., 1.]})

# df_res = gr.ev_df(df=df)
# df_res = model >> gr.ev_df(df=df)

# df_res = model >> gr.ev_nominal()
# df_res = model >> gr.ev_nominal(append=True)

# df_res = gr.eval_grad_fd(model, df_base=df)
# df_res = model >> gr.ev_grad_fd(df_base=df)

# df_res = gr.eval_conservative(model)
df_res = model >> gr.ev_conservative()

print(df_res)
