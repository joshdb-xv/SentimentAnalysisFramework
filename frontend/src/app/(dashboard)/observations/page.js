async function getData() {
  await new Promise((resolve) => setTimeout(resolve, 800));
  return {};
}

export default async function Observations() {
  await getData();

  return (
    <div className="p-8">
      <p>Observations Page</p>
    </div>
  );
}
