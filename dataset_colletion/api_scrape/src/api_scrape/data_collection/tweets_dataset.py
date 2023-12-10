import asyncio
import csv
import dataclasses
import os
import traceback
from typing import Any

from api_scrape.twscrape import AccountsPool, API
from datetime import datetime, timezone, timedelta
from asyncstdlib import enumerate as aenumerate

from api_scrape.data_collection.common import ScrapedTweet, ScrapedTweetWithReason

MOST_OLDEST_UTC_DATE = datetime(2021, 1, 1, tzinfo=timezone.utc)


async def scrape_tweets(username: str, user_id: int, oldest_utc_date: datetime):
    """
        Скачивает твиты не позже заданной даты, выдает НЕ отсортированное по дате

        :param oldest_utc_date: дата в utc, до которой будут скачиваться твиты
        :param username юзер, чьи твиты скачиваем
        :param user_id айдишник этого юзера
    """

    pool = AccountsPool()  # or AccountsPool("path-to.db") - default is `accounts.db`
    await pool.add_account("<name-tag>", "<pass-tag>", "<email>", "<email-pass>")
    print("created account pool")

    await pool.login_all()
    api = API(pool)

    print("created api from pool")

    tweets = []
    tweets_raw = []
    # TODO: переписать на инвариант,
    #  что tweet[x].date > old_date && tweet[x+1].date > tweet[x].date
    #  (от этого юзера)
    # сколько старых твитов подряд ждем, чтобы считать что пора прекращать
    skipped_threshold = 10
    skipped_in_a_row = 0
    async for i, tweet in aenumerate(api.user_tweets_and_replies(user_id)):
        if i % 35 == 1:
            print("Scraped %s tweets from %s" % (i, username))

        tweet_utc_date = tweet.date.astimezone(timezone.utc)
        if tweet_utc_date < oldest_utc_date:
            # to skip pinned tweet
            if i == 0:
                continue
            else:
                skipped_in_a_row += 1
                if skipped_in_a_row > skipped_threshold:
                    break
                else:
                    continue

        skipped_in_a_row = 0

        tweets.append(
            ScrapedTweet(
                tweet.id,
                tweet.user.username,
                tweet_utc_date,
                tweet.rawContent,
                tweet.inReplyToUser.username if tweet.inReplyToUser is not None else None,
                tweet.inReplyToTweetId
            )
        )

        tweets_raw.append(tweet)

    print("Scraped %s tweets from %s" % (len(tweets), username))
    return tweets, tweets_raw


def filter_tweets(implied_username: str, scraped_tweets_not_sorted: list[ScrapedTweet], oldest_utc_date: datetime):
    """
        Фильтрует твиты так,
        чтобы не осталось
        1) реплаев на свои твиты, которые находятся в ветках, созданных другим пользователем
        2) реплаев не на свои твиты
        3) ретвитов
        4) старых твитов

        :param implied_username:
        :param scraped_tweets_not_sorted твиты
        :type scraped_tweets_not_sorted list[ScrapedTweet]
        :param oldest_utc_date дата, после которой твиты старые

        :return два листа: первый с тем что прошло фильтр, второй с тем что НЕ прошло
        :type (list[ScrapedTweet], list[ScrapedTweetWithReason])

        NB: нельзя отредачить твит после реплая на него, поэтому алгоритм:
        проходимся по айдишникам, если айдишник твита, на который реплай все еще в сете, то оставляем
    """

    scraped_tweets_sorted = sorted(scraped_tweets_not_sorted, key=lambda tw: (tw.utc_date, tw.id))

    id_set = set()
    good_tweets = []
    bad_tweets = []

    firstly_filtered_tweets = []
    # с самой давней даты
    for tweet in scraped_tweets_sorted:
        if tweet.id in id_set:
            bad_tweets.append(ScrapedTweetWithReason.from_tweet(tweet, 'Duplicate'))
            continue

        is_retweet = tweet.content.startswith('RT @')
        if is_retweet:
            bad_tweets.append(ScrapedTweetWithReason.from_tweet(tweet, 'Retweet'))
            continue

        if tweet.username != implied_username:
            bad_tweets.append(ScrapedTweetWithReason.from_tweet(tweet, 'WrongAuthor'))
            continue

        id_set.add(tweet.id)
        firstly_filtered_tweets.append(tweet)

    for tweet in firstly_filtered_tweets:
        if tweet.inReplyToUsername not in (None, implied_username):
            bad_tweets.append(ScrapedTweetWithReason.from_tweet(tweet, 'ReplyToAnotherUser'))
            continue

        # могут быть старые твиты, тк твит, на который реплай, поднимается вверх в ленте
        if tweet.utc_date > oldest_utc_date:
            good_tweets.append(tweet)
        else:
            bad_tweets.append(ScrapedTweetWithReason.from_tweet(tweet, 'TooOld'))

    print('Got %s good tweets from %s' % (len(good_tweets), implied_username))
    good_tweets.reverse()
    bad_tweets = sorted(bad_tweets, key=lambda tw: (tw.utc_date, tw.id), reverse=True)

    return good_tweets, bad_tweets


def save_tweets(tweets: list[Any], output_file_name: str):
    if len(tweets) == 0:
        print(f"Nothing to save to {output_file_name}")
        return

    fieldnames = [field.name for field in dataclasses.fields(tweets[0])]
    print(f"Saving tweets as csv with columns {fieldnames}")

    with open(output_file_name, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for tweet in tweets:
            item_dict = dataclasses.asdict(tweet)
            item_dict['utc_date'] = item_dict['utc_date'].isoformat() if 'utc_date' in item_dict \
                else item_dict.get('date', 'None')

            writer.writerow(dataclasses.asdict(tweet))

    print('Saved %s tweets to %s' % (len(tweets), output_file_name))
    return


def parse_already_downloaded(good_tweets_filename: str):
    """
        Получает уже скачанные твиты из файлом и дату самого свежего
        Если файла нет или он пустой, то возвращает OLDEST_UTC_DATE
        Если колонки не соответствуют классу ScrapedTweet, то возвращает OLDEST_UTC_DATE и пустой лист

        :param good_tweets_filename: имя файла с хорошими твитами
        :return newest_tweet_utc_date, previous_good_tweets
        :type (datetime, list[ScrapedTweet])

        NB: считается, что твиты в файле отсортированы, поэтому как самый свежий берется первый
    """

    if not os.path.exists(good_tweets_filename):
        print(f"File {good_tweets_filename} does not exist")
        return MOST_OLDEST_UTC_DATE, []

    expected_fieldnames = [field.name for field in dataclasses.fields(ScrapedTweet)]

    with open(good_tweets_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        if set(reader.fieldnames) != set(expected_fieldnames):
            print(f"{good_tweets_filename} columns don't match ScrapedTweet fields")
            return MOST_OLDEST_UTC_DATE, []

        parsed_tweets = []
        try:
            for row in reader:
                tweet = ScrapedTweet(
                    int(row['id']),
                    row['username'],
                    datetime_from_row(row),
                    row['content'],
                    row['inReplyToUsername'] if row['inReplyToUsername'] else None,
                    int(row['inReplyToTweetId']) if row['inReplyToTweetId'] else None
                )
                parsed_tweets.append(tweet)

            if len(parsed_tweets) == 0:
                return MOST_OLDEST_UTC_DATE, []

            return parsed_tweets[0].utc_date, parsed_tweets

        except Exception as e:
            print(f"Couldn't parse tweets from file: {e}")
            traceback.print_exc()

            return MOST_OLDEST_UTC_DATE, []


def datetime_from_row(csv_row: dict[str, str]):
    return datetime.fromisoformat(csv_row["utc_date"])


async def main():
    # если False, то будет перезаписывать
    should_append_new = True

    twitter_users = [
        {'user': 'elonmusk', 'user_id': 44196397},
        {'user': 'cryptocom', 'user_id': 864347902029709314},
        {'user': 'crypto', 'user_id': 928759224599040001},
        {'user': 'BBCBreaking', 'user_id': 5402612},
        {'user': 'cnnbrk', 'user_id': 428333},
        {'user': 'FoxNews', 'user_id': 1367531},
        {'user': 'financialjuice', 'user_id': 381696140},
        {'user': 'BinanceUS', 'user_id': 1115465940831891457},
        {'user': 'krakenfx', 'user_id': 1399148563},
        {'user': 'federalreserve', 'user_id': 26538229},
        {'user': 'CanadianPM', 'user_id': 14713787},
        {'user': 'govsingapore', 'user_id': 56883209},
        {'user': 'JPN_PMO', 'user_id': 266991549},
        {'user': 'CryptoMichNL', 'user_id': 146008010},
        {'user': 'Bitcoin', 'user_id': 357312062},
        {'user': 'BTCTN', 'user_id': 3367334171},
        {'user': 'NATO', 'user_id': 83795099},
        {'user': 'XHNews', 'user_id': 487118986},
        {'user': 'ForbesInvestor', 'user_id': 17374576},
        {'user': 'Investopedia', 'user_id': 21459600},
        {'user': 'AmerBanker', 'user_id': 26755480},
        {'user': 'khamenei_ir', 'user_id': 27966935},
        {'user': 'WhiteHouse', 'user_id': 1323730225067339784},
        {'user': 'coinbase', 'user_id': 574032254},
        {'user': 'VitalikButerin', 'user_id': 295218901},
        {'user': 'cz_binance', 'user_id': 902926941413453824},
        {'user': 'satyanadella', 'user_id': 20571756},
        {'user': 'TPostMillennial', 'user_id': 896731633704947712},
        {'user': 'Cointelegraph', 'user_id': 2207129125},
        {'user': 'CoinDeskMarkets', 'user_id': 956155022957531137},
        {'user': 'ForexLive', 'user_id': 19399038},
        {'user': 'lopp', 'user_id': 23618940},
        {'user': 'CoinMarketCap', 'user_id': 2260491445},
        {'user': 'Reuters', 'user_id': 1652541},
        {'user': 'TheEconomist', 'user_id': 5988062},
        {'user': 'CNBC', 'user_id': 20402945},
        {'user': 'YahooNews', 'user_id': 7309052},
        {'user': 'EconomicTimes', 'user_id': 39743812},
        {'user': 'CGTNOfficial', 'user_id': 1115874631},
        {'user': 'MailOnline', 'user_id': 15438913},
        {'user': 'FT', 'user_id': 18949452},
        {'user': 'SECGov', 'user_id': 18481050},
        {'user': 'nytimes', 'user_id': 807095},
        {'user': 'SeekingAlpha', 'user_id': 23059499},
        {'user': 'WSJ', 'user_id': 3108351}
    ]

    # assert len(twitter_users) == len(set(twitter_users)), "Not unique list of users"a

    for user in twitter_users:
        username = user['user']
        user_id: int = user['user_id']

        print('\nProcessing %s (id: %s)' % (username, user_id))

        good_tweets_filename = './out/%s_tweets_good.csv' % username
        eliminated_tweets_filename = './out/%s_tweets_eliminated.csv' % username
        raw_tweets_filename = './out/raw/%s_tweets_RAW.csv' % username

        oldest_utc_date = MOST_OLDEST_UTC_DATE
        previous_good_tweets = []

        if should_append_new:
            newest_tweet_utc_date, previous_good_tweets = \
                parse_already_downloaded(good_tweets_filename)

            oldest_utc_date = max(newest_tweet_utc_date + timedelta(seconds=1), oldest_utc_date)
            print('Already downloaded %s good tweets' % len(previous_good_tweets))

        print('Processing not older than: %s' % oldest_utc_date)

        new_user_tweets, new_raw_tweets = await scrape_tweets(username, user_id, oldest_utc_date)

        new_good_tweets, new_bad_tweets = filter_tweets(username, new_user_tweets, oldest_utc_date)

        save_tweets(new_good_tweets + previous_good_tweets, good_tweets_filename)

        # предыдущие отфильтрованные твиты и raw твиты забываются, сохраняются только новые
        save_tweets(new_bad_tweets, eliminated_tweets_filename)
        save_tweets(new_raw_tweets, raw_tweets_filename)

    return


if __name__ == '__main__':
    asyncio.run(main())
