

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```
# execution example
/src
sh run_income.sh > ../reports/income_date.txt
```


```
For xgboost in Mac: brew install libopm
```


### NLTK - SSL Disabled

```
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# nltk.data.path.append('../nltk_data/')
nltk.download("wordnet", "../nltk_data/")
```