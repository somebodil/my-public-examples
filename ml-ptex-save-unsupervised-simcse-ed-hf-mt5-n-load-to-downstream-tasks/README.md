# Notes

Add dataset yourself in `dataset` directory (Too big to upload to git). 
If not specified, program tries to look for `kowikitext_20200920.train`.

# Result

### KLUE STS 

| model            | lr   | Val Score | Test Score |
|------------------|------|-----------|------------|
| google/mt5-small | 1e-3 | 0.9579    | **0.8091** |
| "                | 1e-4 | 0.9466    | 0.762      |
| "                | 1e-5 | 0.9101    | 0.701      |
| "                | 1e-6 | 0.3825    | 0.2288     |

| model                                   | lr   | Val Score | Test Score |
|-----------------------------------------|------|-----------|------------|
| simcse further trained google/mt5-small | 1e-3 | 0.9526    | **0.813**  |
| "                                       | 1e-4 | 0.916     | 0.7001     |
| "                                       | 1e-5 | 0.493     | 0.1457     |
| "                                       | 1e-6 | 0.1373    | -0.05655   |