async function getData() {
  await new Promise((resolve) => setTimeout(resolve, 800));
  return {};
}

export default async function LexicalDictionary() {
  await getData();

  return (
    <div className="p-8">
      <p>LexicalDictionary Page</p>
    </div>
  );
}
