import json
import os
from pathlib import Path

import polars
import psutil
import torch
from google.cloud import storage
from torch.utils.data import DataLoader, TensorDataset

from src import vocabulary
from src.behavegpt import DRO, BehaveGPT, BehaveGPTConfiguration
from src.constant import bucket_name
from src.utils import decompress, is_credential_not_available


def __print_memory(message):
    process = psutil.Process(os.getpid())
    memory = process.memory_info()

    print(f"{message}. real: {memory.rss / 1024 ** 2} MB. total (vms): {memory.vms / 1024 ** 2} MB")


def __count_item(acc, entries: polars.Series) -> list:
    print(entries.list.len())
    return polars.Series([i for i in range(len(entries))])
    
def __get_data_from_cloud_storage(client: storage.Client, bucket: storage.Bucket, folder: str) -> polars.DataFrame:
    folder_data = polars.DataFrame()
    index = 0
    iterator = bucket.list_blobs(prefix=folder, match_glob="**.json")

    for blob in iterator:
        cache_file = Path(blob.name)
        if cache_file.exists():
            rows = polars.read_json(cache_file)
        else:
            content = blob.download_as_bytes(client=client)
            decompressed = decompress(content)
     
            rows = decompressed.splitlines()
            rows = [json.loads(row) for row in rows]
            rows = [row for row in rows if row.get("userId") != ""]
            rows = [
                row
                for row in rows
                if
                    not row.get("eventAction").endswith("loaded")
                    and not row.get("eventAction").startswith("fwf")
            ]

            if len(rows) == 0:
                continue

            rows = polars.DataFrame(rows)
 
        folder_data = polars.concat([folder_data, rows])
        index += 1
 
        if index % 10 == 0:
            print(f"total data at the moment: {len(folder_data)} records")
            __print_memory("current memory when retrieving data from GCS..")

        if not cache_file.exists():
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.touch()
            rows.write_json(cache_file)

        # TODO: to be removed. only for dev
        if len(folder_data) >= 1_000_000:
            break

    return folder_data


def __transpose_dataframe(dataframe: polars.DataFrame) -> polars.DataFrame:
    return (dataframe
        .sort(["eventDate", "eventTime"])
        .group_by(["userId", "brand", "globalEntityId"], maintain_order=True)
        .agg("*")
        .with_columns(
            id=polars.col("eventDate").list.eval(polars.element().cum_count())
        )
        .explode(["id", "eventAction"], empty_as_null=True)
        .pivot(
            on="id",
            index=["userId", "brand", "globalEntityId"],
            values=["eventAction"],
            aggregate_function="first"
        )
    )



def main():
    print("start training...")

    if is_credential_not_available():
        print("no credential found. check https://docs.cloud.google.com/docs/authentication/provide-credentials-adc")
        return

    print("getting available data....")
    __print_memory("starting memory")

    # TODO: to be put in some file and loaded based on user input
    client = storage.Client()
    bucket = client.get_bucket(bucket_or_name=bucket_name, timeout=5)

    iterator = bucket.list_blobs(prefix="temp/", delimiter="/", match_glob="**/")
    list(iterator)  # trigger lazy iterator
    folders = iterator.prefixes
    data: polars.DataFrame = polars.DataFrame() 

    for folder in folders:
        print(f"reading folder: {folder}")
        folder_data = polars.DataFrame()

        cache_file = Path(f"{folder.strip('/')}_cache.json")
        if cache_file.exists():
            print(f"{cache_file} exists. reading it....")
            folder_data = polars.read_json(cache_file)
            __print_memory("current memory after reading from cache")
        else:
            folder_data = __get_data_from_cloud_storage(
                client, bucket, folder
            )

            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.touch()
            folder_data.write_json(cache_file)

        data = polars.concat([data, folder_data])

    filtered = (data
        .filter(~polars.col("eventAction").str.starts_with("joker"))
        .fill_null("<PAD>")
    )
    transformed = __transpose_dataframe(filtered) 

    ## -- event action parts
    actions = (filtered 
        .select("eventAction")
        .group_by(["eventAction"])
        .agg("*")
        .to_series()
        .to_list()
    )
    vocabulary.persist_event_vocabularies(actions)
    event_vocabulary = vocabulary.get_event_vocabularies()
    mapped = filtered.with_columns(polars.col("eventAction").replace(event_vocabulary).alias("eventAction"))
    transposed = __transpose_dataframe(mapped)

    event_actions_tensor = (transposed
            .select(polars.exclude("userId", "brand", "globalEntityId"))
            .fill_null(str(event_vocabulary["<PAD>"]))
            .with_columns(polars.all().cast(polars.Int64))
            .to_torch(return_type="tensor")
    )

    print(event_actions_tensor.size())
    print(event_actions_tensor.type())
    ## -- done event action parts

    ## -- spatial parts
    spaces = (filtered
        .select("globalEntityId")
        .group_by(["globalEntityId"])
        .agg("*")
        .to_series()
        .to_list()
    )
    vocabulary.persist_spatial_vocabularies(spaces)
    space_vocabulary = vocabulary.get_spatial_vocabularies()
    mapped = filtered.with_columns(polars.col("globalEntityId").replace(space_vocabulary).alias("globalEntityId"))
    transposed = __transpose_dataframe(mapped)

    spatial_tensor = (transposed
        .select(polars.col("globalEntityId"))
        .fill_null(str(space_vocabulary["<PAD>"]))
        .with_columns(polars.all().cast(polars.Int64))
        .to_torch(return_type="tensor")
    )

    print(spatial_tensor.size())
    print(spatial_tensor.type())
    ## -- done spatial parts

    ## -- time parts
    # TODO: to convert the time into hour of week (or using time2vec)
    excluded = transformed.select(polars.exclude("userId", "brand", "globalEntityId"))
    temporal_tensor = (excluded
        .with_columns(
            [
                polars.when(polars.col(col).is_not_null())
                    .then(polars.lit(str(i + 1)).alias(col))
                    .otherwise(polars.col(col))
                for i, col in enumerate(excluded.schema)
            ]
        )
        .fill_null("0")
        .with_columns(polars.all().cast(polars.Int64))
        .to_torch(return_type="tensor")
    )

    print(temporal_tensor.size())
    print(temporal_tensor.type())
    ## -- done time parts

    ## -- group parts
    group_data_frame = filtered.select("eventAction")

    group_id_by_event_action = dict(group_data_frame
        .group_by("*")
        .len(name="count")
        .with_columns(polars.sum("count").alias("total"))
        .with_columns(polars.col("count").truediv(polars.col("total")).mul(100).alias("allotment"))
        .with_columns(
            polars
                .when(polars.col("allotment") >= 10)
                .then(polars.lit(1))
                .when(polars.col("allotment") >= 30)
                .then(polars.lit(2))
                .otherwise(polars.lit(3))
                .alias("grouping")
        )
        .select(polars.col("eventAction"), polars.col("grouping"))
        .iter_rows()
    )
    mapped = filtered.with_columns(
        polars.col("eventAction").replace(group_id_by_event_action).alias("eventAction")
    )
    transposed = __transpose_dataframe(mapped)

    group_tensor = (transposed
        .select(polars.exclude("userId", "brand", "globalEntityId"))
        .fill_null(str(event_vocabulary["<PAD>"]))
        .with_columns(polars.all().cast(polars.Int64))
        .to_torch(return_type="tensor")
    )

    print(group_tensor.size())
    print(group_tensor.type())
    ## -- done group part

    _, temporal_token_length = transformed.shape 
    temporal_token_length = temporal_token_length - 3 # removed 3 id columns

    config = BehaveGPTConfiguration(
        num_activity_tokens=len(event_vocabulary),
        activity_padding_index=event_vocabulary["<PAD>"],
        num_spatial_tokens=len(space_vocabulary),
        spatial_padding_index=space_vocabulary["<PAD>"],
        num_temporal_tokens=temporal_token_length,
        temporal_padding_index=0,
        max_seq_len=temporal_token_length
    )
    model = BehaveGPT(config)
    dro = DRO(num_groups=3)  # follow the number of grouping above

    if spatial_tensor.shape[1] == 1:
        spatial_tensor = spatial_tensor.repeat(1, event_actions_tensor.shape[1])

    dataset = TensorDataset(event_actions_tensor, spatial_tensor, temporal_tensor, group_tensor)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    num_epochs = 1

    # -- training mode
    model.train()

    for epoch in range(num_epochs):
        total_epoch_loss = 0.0

        for batch_action, batch_spatial, batch_temporal, batch_group in dataloader:
            optimizer.zero_grad()
    
            logits = model(batch_action, batch_spatial, batch_temporal)
    
            target_logits = logits[:, :-1, :].reshape(-1, config.num_activity_tokens)
            targets = batch_action[:, 1:].reshape(-1)
            shifted_groups = batch_group[:, 1:].reshape(-1)

            loss = dro(target_logits, targets, shifted_groups)
            loss.backward()
            optimizer.step()

            total_epoch_loss = total_epoch_loss + loss.item()

        avg_loss = total_epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Completed | Average Epoch Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "behavegpt.pt")
    torch.save(optimizer.state_dict(), "behavegpt_optimizer.pt")

    # -- end training mode


    # -- example of inference
    model.eval()

    real_user_action = [1, 2, 3]
    real_user_location = [1, 1, 1]
    real_user_time = [1, 1, 2] # in hour to week

    event_index_to_vocab = {value: key for key, value in event_vocabulary}

    with torch.no_grad():
        action_tensor = torch.Tensor(real_user_action).unsqueeze(0)
        location_tensor = torch.Tensor(real_user_location).unsqueeze(0)
        event_action_tensor = torch.Tensor(real_user_time).unsqueeze(0)

        device = next(model.parameters()).device
        action_tensor = action_tensor.to(device)
        location_tensor = location_tensor.to(device)
        event_action_tensor = event_action_tensor.to(device)

        logits = model(action_tensor, location_tensor, event_action_tensor)
        next_step_logits = logits[0, -1, :]

        probabilites = torch.F.softmax(next_step_logits, dim=-1)
        next_top_probabilites, next_top_indices = torch.topk(probabilites, k=5) # only get top 5
        next_top_actions = [event_index_to_vocab[index] for index in next_top_indices]

        print(next_top_probabilites)
        print(next_top_actions)
    # -- end infer mode


if __name__ == "__main__":
    main()
