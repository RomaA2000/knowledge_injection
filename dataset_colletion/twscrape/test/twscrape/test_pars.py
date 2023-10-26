from reply import test_data
from twscrape.utils import to_old_rep
from twscrape.models import Tweet, User

obj = to_old_rep(test_data)

for _, v in obj["tweets"].items():
    print(Tweet.parse(v, obj))


