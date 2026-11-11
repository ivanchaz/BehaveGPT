from datetime import UTC, datetime, timedelta

from google.cloud import bigquery

from src.constant import bucket_name
from src.utils import is_credential_not_available


def main():
    print("start downloading data....")

    if is_credential_not_available():
        print("no credential found. check https://docs.cloud.google.com/docs/authentication/provide-credentials-adc")
        return

    now = datetime.now(UTC)
    date_format = "%d%m%Y"
    formatted_date = now.strftime(date_format)
    previous_hour = (now - timedelta(hours=1)).hour

    entities = ["efood", "baemin_korea"]
    client = bigquery.Client()

    not_like_in_query = """
        and eventAction not like 'joker%'
    """

    for entity in entities:
        output_path = f"gs://{bucket_name}/temp/{formatted_date}_{entity}_{previous_hour}/*.json"

        job = client.query(f"""
          export data options (
            uri='{output_path}',
            format='JSON',
            overwrite=true,
            compression='GZIP'
          ) as
           select
             userId,
             eventAction,
             globalEntityId,
             extract(date from payloadTimestamp) as eventDate,
             extract(time from payloadTimestamp) as eventTime,
             brand
           from `dhh-perseus-live.{entity}.perseus_events`
           where
            timestamp between 
                timestamp(datetime(current_date(),time({previous_hour}, 00, 00)))
                and
                timestamp(datetime(current_date(),time({previous_hour}, 59, 59)))
            and userId != ''
            {not_like_in_query}
        """)
    
        job.result()
        print(f"data stored in: {output_path}")

    print("finished downloading data...")

if __name__ == "__main__":
    main()
