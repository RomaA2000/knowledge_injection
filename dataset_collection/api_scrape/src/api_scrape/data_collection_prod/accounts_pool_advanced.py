from datetime import datetime, timezone

from sortedcontainers import SortedList

from api_scrape.data_collection_prod.exceptions import NewsException, with_news_exception_wrapper
from api_scrape.twscrape import logger
from api_scrape.twscrape.account import Account
from api_scrape.twscrape.utils import find_single_or_throw, utc_datetime_now, throw_if_not_utc_zone


class AccountsPool:

    def __init__(self, given_accounts: list[Account]):
        self.accounts: SortedList[Account] = SortedList(given_accounts, key=lambda acc: acc.last_used)

    @with_news_exception_wrapper
    def _update_last_used(self, username: str, new_last_used: datetime):
        """
        Просто проставляет аккаунту с таким username переданный new_last_used и изменяет его место в очереди
        """
        throw_if_not_utc_zone(new_last_used)
        account_to_update: Account = find_single_or_throw(lambda acc: acc.username == username, self.accounts)
        self.accounts.remove(account_to_update)

        account_to_update.last_used = new_last_used
        self.accounts.add(account_to_update)

    def lock_until(self, username: str, until_utc_datetime: datetime):
        throw_if_not_utc_zone(until_utc_datetime)
        logger.debug(f"Locked account with username: {username} until {until_utc_datetime}")
        # Просто ставим ему такой last_used - он уйдет в конец очереди и будет дальше тех, кто еще не разблокирован
        self._update_last_used(username, until_utc_datetime)

    def get_available(self, number: int) -> list[Account]:
        if number > len(self.accounts):
            raise NewsException(f"Requested {number} accounts, but pool size is {len(self.accounts)}")
        result = []
        current_timestamp = utc_datetime_now()
        for i in range(0, number):
            account = self.accounts[i]
            if account.last_used > current_timestamp:
                logger.warning("Using locked account due to absence of unlocked ones!")
            result.append(account)
        assert len(result) == number

        return result
