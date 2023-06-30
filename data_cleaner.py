import json
import asyncio
import os


class AsyncJSONCleaner:
    def __init__(self):
        self.path = os.sep.join([os.getcwd(), "datasets"])
        self.file_list = [filename for filename in os.listdir(self.path) if
                          ".json" in filename and "cleaned" not in filename]
        os.chdir(self.path)
        asyncio.run(self.main())

    async def clean_line(self, file: dict) -> str:
        for page in file:
            for value, key in file[page].items():
                for v, k in key.items():
                    file[page][value][v] = k.strip().replace('\t', '').replace('\n', '')
        return file

    async def clean_file(self, filename: str):
        file = json.load(open(filename, "r"))
        res = await self.clean_line(file)
        with open("cleaned-" + filename, "a+") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)

    async def main(self):
        tasks = []
        for file in self.file_list:
            tasks.append(self.clean_file(file))
        await asyncio.gather(*tasks)
        print(tasks)


cleaner = AsyncJSONCleaner()
