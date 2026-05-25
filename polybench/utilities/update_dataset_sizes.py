import pprint
import sys

NXL = 6


def parse_file(path: str):
    file = open(path, "r")
    data = {}

    # skip first line
    header = file.readline()
    datasets = header.split()[4:]

    for line in file:
        split = line.split()
        if len(split) == 0:
            continue
        benchmark = split[0]
        nparams = 0
        for i in range(3, len(split)):
            if split[i].isdigit():
                break
            nparams += 1
        data[benchmark] = {
            "category": split[1],
            "datatype": split[2],
            "params": split[3 : 3 + nparams],
            "dataset_sizes": {},
        }

        for i, dataset in enumerate(datasets):
            data[benchmark]["dataset_sizes"][dataset] = tuple(
                map(
                    int,
                    split[
                        3 + nparams + nparams * i : 3 + nparams + nparams * i + nparams
                    ],
                )
            )
    file.close()

    return data


def write_file(data: dict, path: str):
    file = open(path, "w")

    # write header
    file.write("kernel\tcategory\tdatatype\tparams\t")
    for dataset in data[list(data.keys())[0]]["dataset_sizes"]:
        file.write(f"{dataset}\t")
    file.write("\n")

    for benchmark in data:
        file.write(f"{benchmark}\t")
        file.write(f"{data[benchmark]['category']}\t")
        file.write(f"{data[benchmark]['datatype']}\t")
        file.write(f"{' '.join(data[benchmark]['params'])}\t")
        for dataset in data[benchmark]["dataset_sizes"]:
            file.write(
                f"{' '.join(map(str, data[benchmark]['dataset_sizes'][dataset]))}\t"
            )
        file.write("\n")

    file.close()


def main(input_file: str, output_file: str):
    data = parse_file(input_file)

    for i in range(2, NXL + 1):
        for benchmark, benchmark_data in data.items():
            xl_dataset = benchmark_data["dataset_sizes"]["EXTRALARGE"]
            benchmark_data["dataset_sizes"][f"XL{i}"] = tuple(
                map(lambda x, i=i: x * i, xl_dataset)
            )

    pprint.pprint(data)
    write_file(data, output_file)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
