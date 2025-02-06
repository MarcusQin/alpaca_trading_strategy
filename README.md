# IN-CLASS EXERCISE # 1


Welcome to your first in-class exercise in **GitHub classroom**! :octocat:

As you can see, the GitHub classroom repo looks and feels exactly as the repos in your personal GitHub account that you have been using up until this point.

To practice **submitting homework exercises**, follow the steps below from the command line:

1. select a location in your laptop where you'll clone this repo
```
$ cd <your selected location here>
```
2. copy the url for this repo, and
`clone` it to the current location
```
$ git clone <paste-your-url>.git
```
3. create a folder called `src` and add a `test.py` and `test.R` file
```
$ mkdir src
$ touch src/test.py src/test.R
```
(Mac)
```
$ mkdir src
$ echo src/test.py
$ echo src/test.R
```
(Windows)

4. add a line of code to each file
```
$ echo "import pandas as pd" >> src/test.py
$ echo "library(tidyverse)" >> src/test.R
```
(Mac)
```
$ echo import pandas as pd > "src/test.py"
$ echo library(tidyverse) > "src/test.R"
```
(Windows)  

5. take the usual steps to `push` these files back to the repo
```
$ git add src/test.py src/test.R
$ git commit -m "add test files"
$ git push -u origin main
```
