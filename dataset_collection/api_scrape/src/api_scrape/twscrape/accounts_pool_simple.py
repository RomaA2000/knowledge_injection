import asyncio
from datetime import datetime, timezone

from .account import Account
from .logger import logger
from .login import login
from fake_useragent import UserAgent


class AccountsPool:
    @staticmethod
    def guess_delim(line: str):
        lp, rp = tuple([x.strip() for x in line.split("username")])
        return rp[0] if not lp else lp[-1]

    def __init__(self):
        self.accounts: list = []

    async def load_from_file(self, filepath: str, line_format: str):
        raise NotImplementedError("no db")

    async def add_account(
            self,
            username: str,
            password: str,
            email: str,
            email_password: str,
            user_agent: str | None = None,
            proxy: str | None = None,
    ):
        existing_account = self._find_account(username)
        if existing_account:
            logger.debug(f"Account {username} already exists")
            return

        logger.debug(f"Adding account {username}")

        account = Account(
            username=username,
            password=password,
            email=email,
            email_password=email_password,
            user_agent=user_agent or UserAgent().safari,
            active=False,
            locks={},
            stats={},
            headers={},
            cookies={},
            proxy=proxy,
        )

        self.accounts.append(account)

    async def get(self, username: str) -> Account | None:
        found_account = self._find_account(username)

        return found_account.copy() if found_account else None

    async def get_all(self):
        return [Account.copy(acc) for acc in self.accounts]

    async def login(self, account: Account):
        try:
            await login(account)
            logger.info(f"Logged in to {account.username} successfully")
        except Exception as e:
            logger.error(f"Error logging in to {account.username}: {e}")
        finally:
            self._remove_account(account.username)
            self.accounts.append(account)

    async def login_all(self):
        for i, acc in enumerate(self.accounts, start=1):
            logger.info(f"[{i}/{len(self.accounts)}] Logging in {acc.username} - {acc.email}")
            await self.login(acc)

    async def set_active(self, username: str, active: bool):
        acc = self._find_account(username)
        if not acc:
            raise ValueError(f"Account {username} not found")

        acc.active = active

    async def lock_until(self, username: str, queue: str, unlock_at: int, req_count: int = 0):
        acc = self._find_account(username)
        if not acc:
            raise ValueError(f"Account {username} not found")

        acc.locks[queue] = datetime.fromtimestamp(unlock_at, tz=timezone.utc)

    async def unlock(self, username: str, queue: str, req_count=0):
        acc = self._find_account(username)
        if not acc:
            raise ValueError(f"Account {username} not found")

        del acc.locks[queue]

        acc.stats[queue] = acc.stats.get(queue, 0) + req_count

    async def get_for_queue(self, queue: str):
        time_now = datetime.now(tz=timezone.utc)

        available_acc = next(
            (
                acc for acc in self.accounts
                if acc.active and (queue not in acc.locks or acc.locks[queue] < time_now)
            ),
            None
        )

        if not available_acc:
            return None

        await self.lock_until(available_acc.username, queue, int(time_now.timestamp()) + 15 * 60)

        return available_acc

    async def get_for_queue_or_wait(self, queue: str) -> Account:
        while True:
            account = await self.get_for_queue(queue)
            if not account:
                logger.debug(f"No accounts available for queue '{queue}' (sleeping for 5 sec)")
                await asyncio.sleep(5)
                continue

            logger.debug(f"Using account {account.username} for queue '{queue}'")
            return account

    async def stats(self):
        raise NotImplementedError("no stats")

    async def accounts_info(self):
        items = await self.get_all()

        old_time = datetime(1970, 1, 1).replace(tzinfo=timezone.utc)
        items = sorted(items, key=lambda x: x["username"].lower())
        items = sorted(items, key=lambda x: x["last_used"] or old_time, reverse=True)
        items = sorted(items, key=lambda x: x["total_req"], reverse=True)
        items = sorted(items, key=lambda x: x["active"], reverse=True)
        return items

    def _remove_account(self, username: str):
        self.accounts = list(filter(lambda acc: acc.username == username, self.accounts))

    def _find_account(self, username: str) -> Account:
        return next((acc for acc in self.accounts if acc.username == username), None)
