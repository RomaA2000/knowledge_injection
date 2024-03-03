import csv
import dataclasses
import datetime
import os
from datetime import datetime
from typing import List, Optional, Tuple
from typing import Dict

from itertools import groupby
from api_scrape.twscrape import logger

from api_scrape.twscrape import Tweet
from api_scrape.twscrape.utils import first_index_of


# NewsResultSender
# - batch(list[Tweet_raw]) - батч твитов, который скачал инстанс главного сервера
# - разбить по аккаунтам-авторам, и дописать в соответствующие файлы.
# - В файлах содержится {**ScrapedTweet, **download_time_data}
# - Проверить, что нету гэпа: самый новый уже скачанный твит в констекте инстанса (не аккаунта), содержится в батче. Если нет, то это фейл

class NewsSenderTmp:
    def __init__(self, expected_usernames: list[str], directory_to_save_path: str):
        self.expected_usernames: list[str] = expected_usernames
        if os.path.exists(directory_to_save_path):
            raise Exception(f"Directory with tweets \"{directory_to_save_path}\" already exists, please delete it")
        os.makedirs(directory_to_save_path)
        self.username_to_file_name: Dict[str, str] = {username: f'{directory_to_save_path}/{username}.csv' for username
                                                      in
                                                      self.expected_usernames}
        self.last_tweet_id: Optional[int] = None
        self.csv_file_with_tweets_headers = [field.name for field in dataclasses.fields(Tweet)]
        self.save_time_column_name = 'download_time_utc'
        self.csv_field_names = [field.name for field in dataclasses.fields(Tweet)] + [self.save_time_column_name]

    # кик реплаи
    # кик юзеров которые не в expected_usernames
    def send_news(self, tweets_batch: list[Tweet]):
        current_datetime = datetime.utcnow()
        filtered_by_accounts_sorted_tweets = sorted(self._filter_tweets(tweets_batch), key=_tweets_sorting_def)
        last_batch_newest_tweet_index = first_index_of(lambda tweet: tweet.id == self.last_tweet_id,
                                                       filtered_by_accounts_sorted_tweets)
        if self.last_tweet_id is not None and last_batch_newest_tweet_index == -1:
            # Если не нашли самый новый твит из прошлого батча просто пишем еррор в консоль и записываем все
            logger.error(f"No last tweet with id: {self.last_tweet_id} found.")
        else:
            # Иначе берем все твиты после даты самого нового твита из прошлого батча
            filtered_by_accounts_sorted_tweets = filtered_by_accounts_sorted_tweets[last_batch_newest_tweet_index + 1:]

        if len(filtered_by_accounts_sorted_tweets) == 0:
            return
        # Обновляем ID самого нового твита из батча
        self.last_tweet_id = filtered_by_accounts_sorted_tweets[-1].id

        # Делаем мапу из имени пользователя в список твиттов по нему
        grouped_data: Dict[str, List[Tweet]] = {
            k: list(v) for k, v in groupby(filtered_by_accounts_sorted_tweets, key=lambda t: t.user.username)
        }

        for username, tweets in grouped_data.items():
            # Я не уверен что при groupby не нарушился порядок, поэтому сортирую еще раз
            sorted_tweets_for_exact_account = sorted(tweets, key=_tweets_sorting_def)
            file_to_append_tweets_name = self.username_to_file_name[username]
            self._save_tweets_to_file(
                tweets=sorted_tweets_for_exact_account,
                file_to_append_name=file_to_append_tweets_name,
                save_time=current_datetime
            )

    def _save_tweets_to_file(self, tweets: List[Tweet], file_to_append_name: str, save_time: datetime):
        if len(tweets) == 0:
            return
        append_header = not os.path.exists(file_to_append_name)
        with open(file_to_append_name, mode='a', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=self.csv_field_names)
            if append_header:
                writer.writeheader()
            for tweet in tweets:
                tweet_as_dict = dataclasses.asdict(tweet)
                tweet_as_dict[self.save_time_column_name] = save_time
                writer.writerow(tweet_as_dict)

    def _filter_tweets(self, batch: List[Tweet]) -> List[Tweet]:
        """ Фильтрует твиты, оставляя только из аккаунтов, которые нас интересуют
        и те, которые НЕ реплаи
        """
        result = []
        for tweet in batch:
            if tweet.user.username in self.expected_usernames:
                result.append(tweet)
            else:
                logger.warning(f'Tweet from an unexpected account {tweet}')
        return result


def _tweets_sorting_def(tweet: Tweet) -> Tuple[datetime, str]:
    return tweet.date, tweet.user.username
