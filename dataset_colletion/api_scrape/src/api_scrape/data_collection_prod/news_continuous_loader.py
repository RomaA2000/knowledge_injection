import asyncio
import threading
from asyncio import Task
from datetime import timedelta, datetime

from apscheduler.schedulers.background import BlockingScheduler

from api_scrape.data_collection_prod.accounts_pool_advanced import AccountsPool
from api_scrape.data_collection_prod.exceptions import TweetLimitReachedException, UserBannedException, NewsException, \
    LoginException
from api_scrape.data_collection_prod.news_sender import NewsSenderTmp
from api_scrape.data_collection_prod.twitter_requests import login_agent, download_batch_search_tweets
from api_scrape.twscrape import Account, Tweet
from api_scrape.twscrape.logger import logger
from api_scrape.twscrape.utils import utc_datetime_now


class NewsContinuousLoader:
    def __init__(self, agents_pool: AccountsPool, target_users: list[str]):
        if len(target_users) > 23:
            raise ValueError(f"Too many target users ({len(target_users)}), max is 23")
        if len(agents_pool.accounts) == 0:
            raise ValueError("No agents provided")

        self._agents_pool = agents_pool
        self._target_users = target_users

        self._ready_agents = []
        self._news_sender = NewsSenderTmp(target_users)

        self._lock = threading.Lock()

    def run(self):
        phase_scheduler = BlockingScheduler()
        phase_scheduler.add_job(self._phase1, 'cron', second=30)
        phase_scheduler.add_job(self._phase2, 'cron', second=55)

        phase_scheduler.start()

    def _phase1(self):
        logger.debug("Starting phase 1")

        with self._lock:
            agents_candidates = self._agents_pool.get_available(3)
            agents_passed = []

            tasks_to_agents = {asyncio.create_task(self._get_batch_with_login_task(agent)): agent for agent in agents_candidates}
            done, _ = asyncio.wait(tasks_to_agents.keys(), timeout=15, return_when=asyncio.ALL_COMPLETED)

            for task in done:
                agent = tasks_to_agents[task]

                if self._process_done_task(task, agent.username):
                    logger.debug(f"Agent {tasks_to_agents[task].username} completed phase 1")
                    agents_passed.append(agent)

            if len(agents_passed) == 0:
                logger.error("No agents completed phase 1")

            self._ready_agents = agents_passed

    def _phase2(self):
        logger.debug("Starting phase 2")

        with self._lock:
            if len(self._ready_agents) == 0:
                logger.error("No ready agents for phase 2")
                return

            tasks_to_agents = {asyncio.create_task(self._get_batch_task(agent)): agent for agent in self._ready_agents}
            done, _ = asyncio.wait(tasks_to_agents.keys(), timeout=3, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                if self._process_done_task(task, tasks_to_agents[task].username):
                    logger.debug(f"Agent {tasks_to_agents[task].username} completed phase 2")
                    return

            logger.error("No agents completed phase 2")

    async def _get_batch_with_login_task(self, agent: Account) -> list[Tweet]:
        await login_agent(agent)
        return await self._get_batch_task(agent)

    async def _get_batch_task(self, agent: Account) -> list[Tweet]:
        return await download_batch_search_tweets(agent, self._target_users)

    def _send_batch(self, tweets_batch: list[Tweet]):
        self._news_sender.send_news(tweets_batch)

    # True, если без эксепшена
    def _process_done_task(self, task: Task, agent_username: str) -> bool:
        try:
            batch = task.result()
            self._send_batch(batch)
            logger.debug(f"{agent_username} sent batch of {len(batch)} tweets")
            return True

        except Exception as e:
            self._process_task_exception(agent_username, e)
            return False

    def _process_task_exception(self, agent_username: str, e: Exception):
        blocking_date = utc_datetime_now() + timedelta(minutes=15)

        if isinstance(e, TweetLimitReachedException):
            reset_timestamp = e.limit_timestamp
            if reset_timestamp and reset_timestamp != 0:
                blocking_date = datetime.utcfromtimestamp(reset_timestamp) + timedelta(seconds=5)

            logger.error(f"Tweets limit reached for agent {agent_username}")

        elif isinstance(e, UserBannedException):
            blocking_date = utc_datetime_now() + timedelta(minutes=30)
            logger.error(f"User {agent_username} was banned")

        elif isinstance(e, LoginException):
            blocking_date = utc_datetime_now() + timedelta(minutes=30)
            logger.error(f"Login {agent_username} failed")

        elif isinstance(e, NewsException):
            blocking_date = utc_datetime_now() + timedelta(minutes=10)
            logger.error(f"Error while getting tweets for agent {agent_username}: {e}")

        else:
            logger.error(f"UNEXPECTED error while getting tweets for agent {agent_username}: {e}")

        logger.info(f"Blocking agent {agent_username} until {blocking_date}")
        self._agents_pool.lock_until(agent_username, blocking_date)

