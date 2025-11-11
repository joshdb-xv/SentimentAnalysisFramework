import DictionaryUpload from "./DictionaryUpload";
import KeywordsUpload from "./KeywordsUpload";

export default function FileUploadSection({
  dictionaryFile,
  keywordsFile,
  dictionaryInfo,
  keywordsInfo,
  uploadingDict,
  uploadingKeywords,
  loadingKeywords,
  onDictionaryUpload,
  onKeywordsLoad,
  onKeywordsUpload,
}) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <DictionaryUpload
        file={dictionaryFile}
        info={dictionaryInfo}
        uploading={uploadingDict}
        onUpload={onDictionaryUpload}
      />

      <KeywordsUpload
        file={keywordsFile}
        info={keywordsInfo}
        uploading={uploadingKeywords}
        loading={loadingKeywords}
        onLoad={onKeywordsLoad}
        onUpload={onKeywordsUpload}
      />
    </div>
  );
}
