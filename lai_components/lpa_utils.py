

def output_with_video_prediction(lines) -> dict:
  """outputs/2022-07-04/17-28-54/test_vid_heatmap.csv"""
  outputs = {}
  for l in lines:
    l_split = l.strip().split("/")
    value = l_split[-1]
    key = "/".join(l_split[-3:-1])
    outputs[key] = value
  return(outputs)  