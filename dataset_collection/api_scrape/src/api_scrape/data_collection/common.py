import dataclasses
import datetime
import typing


@dataclasses.dataclass
class JSONTweet:
    rawJSON: str


@dataclasses.dataclass
class ScrapedTweet:
    id: int
    username: str
    utc_date: datetime.datetime
    content: str
    inReplyToUsername: typing.Optional[str] = None
    inReplyToTweetId: typing.Optional[int] = None


@dataclasses.dataclass
class ScrapedTweetWithRetweetFlag:
    scrapedTweet: ScrapedTweet
    isRetweet: bool


@dataclasses.dataclass
class ScrapedTweetWithReason(ScrapedTweet):
    reason: str = 'N/A'

    @classmethod
    def from_tweet(cls, tweet: ScrapedTweet, reason: str):
        return cls(
            tweet.id,
            tweet.username,
            tweet.utc_date,
            tweet.content,
            tweet.inReplyToUsername,
            tweet.inReplyToTweetId,
            reason
        )
