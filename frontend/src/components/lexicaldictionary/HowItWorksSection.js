export default function HowItWorksSection() {
  return (
    <div className="bg-white shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] p-8 rounded-2xl">
      <div className="flex flex-row items-center gap-4 mb-6">
        <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
        <p className="font-medium text-xl text-[#1E293B]">HOW IT WORKS</p>
      </div>

      <div className="space-y-6">
        <p className="text-gray-mid">
          This tool processes Filipino/Cebuano dictionaries to generate
          VADER-compatible sentiment lexicons using FastText embeddings.
        </p>

        <div className="bg-bluish-white p-6 rounded-xl border-l-4 border-primary">
          <h3 className="font-semibold text-primary-dark mb-4">
            Sample Processing: Word "init"
          </h3>

          <div className="space-y-4 text-sm">
            {/* Step 1 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center font-bold">
                1
              </div>
              <div className="flex-1">
                <p className="font-semibold text-primary-dark mb-1">
                  Word Cleaning
                </p>
                <p className="text-gray-mid">
                  The word is converted to lowercase and hyphens are removed for
                  simpler form.
                </p>
                <div className="mt-2 p-3 bg-white rounded-lg border border-bluish-gray font-mono text-xs">
                  <span className="text-gray">"Init"</span> →{" "}
                  <span className="text-primary font-semibold">"init"</span>
                </div>
              </div>
            </div>

            {/* Step 2 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center font-bold">
                2
              </div>
              <div className="flex-1">
                <p className="font-semibold text-primary-dark mb-1">
                  Climate Keyword Detection
                </p>
                <p className="text-gray-mid">
                  The definition is checked against climate keywords (weather,
                  temperature, climate, etc.) to determine if it's
                  climate-related.
                </p>
                <div className="mt-2 p-3 bg-white rounded-lg border border-bluish-gray">
                  <p className="text-xs text-gray-mid mb-1">Definition:</p>
                  <p className="font-mono text-xs">"Hot weather condition"</p>
                  <p className="text-xs text-primary font-semibold mt-2">
                    ✓ Climate-related: Contains "weather"
                  </p>
                </div>
              </div>
            </div>

            {/* Step 3 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center font-bold">
                3
              </div>
              <div className="flex-1">
                <p className="font-semibold text-primary-dark mb-1">
                  Base Sentiment Score
                </p>
                <p className="text-gray-mid">
                  Assign base score from sentiment label. Climate-related words
                  get stronger scores.
                </p>
                <div className="mt-2 p-3 bg-white rounded-lg border border-bluish-gray">
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <p className="text-gray-mid mb-1">Non-climate:</p>
                      <p className="font-mono">
                        Positive:{" "}
                        <span className="text-blue font-semibold">+2.5</span>
                      </p>
                      <p className="font-mono">
                        Negative:{" "}
                        <span className="text-red font-semibold">-2.5</span>
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-mid mb-1">Climate-related:</p>
                      <p className="font-mono">
                        Positive:{" "}
                        <span className="text-blue font-semibold">+3.25</span>
                      </p>
                      <p className="font-mono">
                        Negative:{" "}
                        <span className="text-red font-semibold">-3.25</span>
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Step 4 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center font-bold">
                4
              </div>
              <div className="flex-1">
                <p className="font-semibold text-primary-dark mb-1">
                  FastText Semantic Intensity
                </p>
                <p className="text-gray-mid">
                  Calculate semantic similarity to anchor words using FastText
                  embeddings to adjust intensity (0.6x to 1.4x).
                </p>
                <div className="mt-2 p-3 bg-white rounded-lg border border-bluish-gray">
                  <div className="text-xs space-y-2">
                    <p className="text-gray-mid">
                      Negative anchors: dili, dautan, grabe, lisud, makuyaw
                    </p>
                    <p className="font-mono">
                      Avg similarity:{" "}
                      <span className="font-semibold">0.65</span>
                    </p>
                    <p className="font-mono">
                      Intensity multiplier:{" "}
                      <span className="font-semibold text-primary">
                        0.6 + (0.65 × 0.8) = 1.12
                      </span>
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Step 5 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center font-bold">
                5
              </div>
              <div className="flex-1">
                <p className="font-semibold text-primary-dark mb-1">
                  Final Score Calculation
                </p>
                <p className="text-gray-mid">
                  Multiply base score by intensity multiplier for the final
                  sentiment score.
                </p>
                <div className="mt-2 p-3 bg-gradient-to-r from-primary/10 to-primary/5 rounded-lg border-2 border-primary">
                  <p className="font-mono text-sm">
                    <span className="text-gray-mid">
                      Base score × Intensity =
                    </span>
                    <span className="font-bold text-primary ml-2">
                      -3.25 × 1.12 = -3.64
                    </span>
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-blue/10 p-4 rounded-xl border border-blue">
          <p className="text-sm text-blue flex items-start gap-2">
            <span className="text-xl">💡</span>
            <span>
              <strong>Duplicate Handling:</strong> If multiple definitions exist
              for the same word, the system prioritizes climate-related
              definitions and keeps simpler word forms (without hyphens).
            </span>
          </p>
        </div>
      </div>
    </div>
  );
}
