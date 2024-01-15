class NewsException(Exception):
    def __init__(self, message: str, internal_cause: Exception = None):
        super().__init__(message)
        self.internal_cause = internal_cause

    pass


class LoginException(NewsException):
    pass


class UserBannedException(NewsException):
    pass


class NotFoundException(NewsException):
    pass


class TweetLimitReachedException(NewsException):
    def __init__(self, limit_timestamp: int, message: str = None, internal_cause: Exception = None):
        super().__init__(message, internal_cause)
        self.limit_timestamp = limit_timestamp

    pass


class GapBetweenBatchesException(NewsException):
    pass


def with_news_exception_wrapper(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise NewsException(str(e), e)

    return wrapper
