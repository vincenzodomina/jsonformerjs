type AnyObject = { [key: string]: any };
type NestedObject = AnyObject | any[] | string;

function highlightValues(value: NestedObject): void {
  function recursivePrint(obj: NestedObject, indent: number = 0, isLastElement: boolean = true): void {
    if (typeof obj === "object" && obj !== null && !Array.isArray(obj)) {
      console.log("{");
      const keys = Object.keys(obj);
      const lastKey = keys[keys.length - 1];
      for (const key of keys) {
        const val = obj[key];
        process.stdout.write(`${" ".repeat(indent + 2)}${key}: `);
        recursivePrint(val, indent + 2, key === lastKey);
      }
      process.stdout.write(`${" ".repeat(indent)}}${!isLastElement ? ",\n" : "\n"}`);
    } else if (Array.isArray(obj)) {
      console.log("[");
      for (let index = 0; index < obj.length; index++) {
        const val = obj[index];
        process.stdout.write(`${" ".repeat(indent + 2)}`);
        recursivePrint(val, indent + 2, index === obj.length - 1);
      }
      process.stdout.write(`${" ".repeat(indent)}]${!isLastElement ? ",\n" : "\n"}`);
    } else {
      if (typeof obj === "string") {
        obj = `"${obj}"`;
      }
      process.stdout.write(`\x1b[32m${obj.toString()}\x1b[0m` + (!isLastElement ? ",\n" : "\n"));
    }
  }

  recursivePrint(value);
}

export default highlightValues;