# 3D Scene Flow Submission Format

The evaluation expects a zip archive of pandas DataFrames stored as feather files, one for each example. The unzipped directory must have the format:
- <test_log_1>/
  - <test_timestamp_ns_1>.feather
  - <test_timestamp_ns_2>.feather
  - ...
- <test_log_2>/
- ...

Each feather file should contain your flow predictions for the indices returned by `get_eval_indices` in the format:

- `flow_tx_m`: x-component of the flow in the first sweeps's egovehicle reference frame.
- `flow_ty_m`: y-component of the flow in the first sweeps's egovehicle reference frame.
- `flow_tz_m`: z-component of the flow in the first sweeps's egovehicle reference frame.


For example the first log in the test set is `0c6e62d7-bdfa-3061-8d3d-03b13aa21f68` and the first timestamp is `315971435999927221`, so there should be a folder and file in the archive of the form: `0c6e62d7-bdfa-3061-8d3d-03b13aa21f68/315971435999927221.feather`. That fill should look like this:
```
       flow_tx_m  flow_ty_m  flow_tz_m
0      -0.699219   0.002869   0.020233
1      -0.699219   0.002790   0.020493
2      -0.699219   0.002357   0.020004
3      -0.701172   0.001650   0.013390
4      -0.699219   0.002552   0.020187
...          ...        ...        ...
68406  -0.703613  -0.001801   0.002373
68407  -0.704102  -0.000905   0.002567
68408  -0.704590  -0.001390   0.000397
68409  -0.704102  -0.001608   0.002283
68410  -0.704102  -0.001619   0.002207
```

The script `package_submission.py` will create the zip archive for you and validate its structure. Lastly, submit this file to the competition leaderboard!
