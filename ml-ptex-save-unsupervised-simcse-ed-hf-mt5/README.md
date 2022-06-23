# Notes

Add dataset yourself in `dataset` directory (Too big to upload to git). 
If not specified, program tries to look for `kowikitext_20200920.train`.

Program will save checkpoint based on your `model_name`, every 5000 iteration of mini-batch, at `checkpoint/{model_state_name}`.
