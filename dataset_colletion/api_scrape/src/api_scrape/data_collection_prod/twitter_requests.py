from api_scrape.data_collection_prod.exceptions import LoginException, TweetLimitReachedException, UserBannedException, \
    NewsException
from api_scrape.twscrape.account import Account
from api_scrape.twscrape.api import SEARCH_FEATURES
from api_scrape.twscrape.constants import GQL_URL, SEARCH_TIMELINE_GQL_PATH, GQL_FEATURES
from api_scrape.twscrape.login import login
from api_scrape.twscrape.logger import logger
from api_scrape.twscrape.models import Tweet
from api_scrape.twscrape.utils import to_old_rep, encode_params

import httpx
import random


async def login_agent(agent: Account):
    """
    Логинит агента, записывает в него данные, чтобы можно было делать запрос
    Кидает ексепшн если не получилось
    """

    try:
        # не хотим кешировать логин, явно логинимся каждый раз
        agent.active = False
        await login(agent, fresh=False, suppress_http_status_error=False, close_client=True)
    except Exception as e:
        raise LoginException(f"Error logging in to {agent.username}: {e}")


async def download_batch_search_tweets(account: Account, target_users: list[str]) -> list[Tweet]:
    """
    скачивает 1 батч твитов (то что получается после =1 запроса к апи)
    НЕ ИМЕЕТ СМЫСЛА БЕЗ ПРЕДВАРИТЕЛЬНОГО ЛОГИНА

    :param account: аккаунт, с которого скачиваем
    :param target_users: список id пользователей, чьи твиты скачиваем
    """

    url = f"{GQL_URL}/{SEARCH_TIMELINE_GQL_PATH}"
    params = _build_params(target_users)
    logger.debug(f"Requesting {url} with params: {params}")

    client = account.make_client()
    logger.debug(f"Created client for {account.username} - {account.email}")

    result = []
    try:
        response = await client.get(url, params=params)
        limit_remaining = response.headers.get("x-rate-limit-remaining", "unknown")
        limit_limit = response.headers.get("x-rate-limit-limit", "unknown")
        logger.info(f"Got batch from {account.username} (limit {limit_remaining}/{limit_limit})")

        received_json = response.json()

        if "errors" in received_json:
            raise NewsException(f"Got errors in batch query: {received_json['errors']}")

        batch_obj = to_old_rep(received_json)
        for x in batch_obj["tweets"].values():
            result.append(Tweet.parse(x, batch_obj))

        logger.info(f"Got {len(result)} raw tweets with {account.username} agent")
        return result

    except httpx.HTTPStatusError as e:
        rep = e.response

        if rep.status_code == 429:
            reset_ts = int(rep.headers.get("x-rate-limit-reset", 0))
            raise TweetLimitReachedException(reset_ts, f"Reached limit for {account.username} agent", e)

        elif rep.status_code in (401, 403):
            raise UserBannedException(f"Agent {account.username} was banned", e)

        else:
            logger.warning(f"HTTP Error {rep.status_code} {e.request.url}\n{rep.text}")
            raise NewsException("Batch request unknown error", e)
    except NewsException as e:
        raise e
    except Exception as e:
        logger.warning(f"Unknown error in batch request {e}")
        raise NewsException("Batch request unknown error", e)

    finally:
        await client.aclose()
        logger.debug(f"Closed client for {account.username} agent")


def _build_params(target_users: list[str]) -> dict[str, str]:
    q = _build_search_query(target_users)
    variables = {
        "rawQuery": q,
        "count": 30,
        "product": "Latest",
        "querySource": "typed_query"
    }
    params = {
        "variables": variables,
        "features": {**GQL_FEATURES, **SEARCH_FEATURES},
        "fieldToggles": {"withArticleRichContentState": False}
    }

    return encode_params(params)


def _build_search_query(target_users: list[str]) -> str:
    shuffled_users = target_users.copy()
    random.shuffle(shuffled_users)

    accounts_joined = " OR ".join(map(lambda acc_name: f"from:{acc_name}", shuffled_users))
    search_query: str = f'({accounts_joined})'
    if len(search_query) > 512:
        raise NewsException("Too many users in search query")

    return search_query
