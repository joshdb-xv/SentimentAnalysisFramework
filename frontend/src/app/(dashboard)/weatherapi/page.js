async function getData() {
  await new Promise((resolve) => setTimeout(resolve, 800));
  return {};
}

export default async function WeatherAPI() {
  await getData();

  return (
    <div className="p-8">
      <p>WeatherAPI Page</p>
    </div>
  );
}
