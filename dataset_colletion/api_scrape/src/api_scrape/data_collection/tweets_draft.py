import requests
import asyncio
from api_scrape.twscrape import AccountsPool, API


def send_search_timeline_request():
    url = "https://cdn.syndication.twimg.com/tweet-result"

    querystring = {"id": "1652193613223436289", "lang": "en"}

    payload = ""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/114.0",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Origin": "https://platform.twitter.com",
        "Connection": "keep-alive",
        "Referer": "https://platform.twitter.com/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "cross-site",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
        "TE": "trailers"
    }

    response = requests.request("GET", url, data=payload, headers=headers, params=querystring)

    print(response.text)


async def twscrape_sample():
    pool = AccountsPool()  # or AccountsPool("path-to.db") - default is `accounts.db`

    # log in to all new accounts
    await pool.login_all()

    api = API(pool)

    print("api prepared")

    bibaid = await api.user_by_login("<name-tag>")
    print(bibaid)
    # i = 0
    # async for tweet in api.user_tweets_and_replies(44196397):
    #     print(tweet.id, tweet.user.username, tweet.rawContent, tweet.date)  # tweet is `Tweet` object
    #     i += 1
    # print("Received %d tweets" % i)


def main():
    send_search_timeline_request()


if __name__ == "__main__":
    asyncio.run(twscrape_sample())

# if __name__ == '__main__':
#     main()

# UserTweets
# https://twitter.com/i/api/graphql/QqZBEqganhHwmU9QscmIug/UserTweets?variables=\
#     {
#         "userId":"1652541",
#         "count":20,
#         "includePromotedContent":true,
#         "withQuickPromoteEligibilityTweetFields":true,
#         "withVoice":true,"withV2Timeline":true
#     }&features=\
#     {
#         "rweb_lists_timeline_redesign_enabled":true,
#         "responsive_web_graphql_exclude_directive_enabled":true,
#         "verified_phone_label_enabled":false,
#         "creator_subscriptions_tweet_preview_api_enabled":true,
#         "responsive_web_graphql_timeline_navigation_enabled":true,
#         "responsive_web_graphql_skip_user_profile_image_extensions_enabled":false,
#         "tweetypie_unmention_optimization_enabled":true,
#         "responsive_web_edit_tweet_api_enabled":true,
#         "graphql_is_translatable_rweb_tweet_is_translatable_enabled":true,
#         "view_counts_everywhere_api_enabled":true,
#         "longform_notetweets_consumption_enabled":true,
#         "responsive_web_twitter_article_tweet_consumption_enabled":false,
#         "tweet_awards_web_tipping_enabled":false,
#         "freedom_of_speech_not_reach_fetch_enabled":true,
#         "standardized_nudges_misinfo":true,
#         "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled":true,
#         "longform_notetweets_rich_text_read_enabled":true,
#         "longform_notetweets_inline_media_enabled":true,
#         "responsive_web_media_download_video_enabled":false,
#         "responsive_web_enhance_cards_enabled":false
#     }&fieldToggles={"withArticleRichContentState":false}
