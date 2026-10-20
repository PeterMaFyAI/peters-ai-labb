/**
 * A hand-authored weighted grammar, expanded into a complete prefix database.
 * Weights are teaching estimates, not measurements from a trained model.
 * Every reachable prefix has its own five-entry table; generation only looks up
 * that table. Shared grammar rules keep the source reviewable and editable.
 */
export const ANSWER_LENGTH = 5;

export const PROMPTS = [
  { id: "capital", label: "En faktafråga", text: "Vilken är Sveriges huvudstad?", start: "capital" },
  { id: "sum", label: "En räknefråga", text: "Vad är 2 + 2?", start: "sum" },
  { id: "story", label: "En berättelse", text: "Fortsätt meningen: En robot öppnade dörren och", start: "story" }
] as const;

export type PromptId = (typeof PROMPTS)[number]["id"];
export interface Candidate { token: string; probability: number }
type Choice = readonly [token: string, weight: number, next: string];
type Rule = readonly Choice[];

// Each rule has exactly five distinct whole words, numbers or punctuation marks.
const rules: Record<string, Rule> = {
  capital: [["Stockholm", 68, "stockholm"], ["Sveriges", 18, "swedens"], ["Huvudstaden", 9, "capitalNoun"], ["Det", 4, "itCapital"], ["Jag", 1, "iCapital"]],
  stockholm: [["är", 65, "isCapital"], ["ligger", 12, "located"], [",", 10, "cityApposition"], [".", 10, "sentenceStart"], ["!", 3, "sentenceStart"]],
  isCapital: [["Sveriges", 72, "capitalObject"], ["landets", 15, "capitalObject"], ["en", 7, "cityAdjective"], ["också", 4, "possessiveCountry"], ["väl", 2, "possessiveCountry"]],
  possessiveCountry: [["Sveriges", 72, "capitalObject"], ["landets", 17, "capitalObject"], ["vår", 6, "capitalObject"], ["en", 3, "cityAdjective"], ["den", 2, "swedishAdjective"]],
  capitalObject: [["huvudstad", 94, "ending"], ["största", 3, "cityNoun"], ["politiska", 1, "politicalNoun"], ["vackra", 1, "cityNoun"], ["kulturella", 1, "politicalNoun"]],
  swedens: [["huvudstad", 94, "capitalVerb"], ["största", 3, "cityThenVerb"], ["politiska", 1, "politicalThenVerb"], ["nuvarande", 1, "capitalThenVerb"], ["vackra", 1, "capitalThenVerb"]],
  capitalThenVerb: [["huvudstad", 94, "capitalVerb"], ["storstad", 2, "capitalVerb"], ["stad", 2, "capitalVerb"], ["metropol", 1, "capitalVerb"], ["centralort", 1, "capitalVerb"]],
  cityThenVerb: [["stad", 75, "capitalVerb"], ["kommun", 12, "capitalVerb"], ["storstadsregion", 7, "capitalVerb"], ["tätort", 4, "capitalVerb"], ["metropol", 2, "capitalVerb"]],
  politicalThenVerb: [["centrum", 80, "capitalVerb"], ["nav", 8, "capitalVerb"], ["hjärta", 6, "capitalVerb"], ["mittpunkt", 4, "capitalVerb"], ["huvudstad", 2, "capitalVerb"]],
  capitalVerb: [["är", 85, "cityAnswer"], ["heter", 10, "cityAnswer"], ["kallas", 3, "cityAnswer"], ["blev", 1, "cityAnswer"], ["förblir", 1, "cityAnswer"]],
  cityAnswer: [["Stockholm", 98, "ending"], ["Göteborg", 0.7, "ending"], ["Malmö", 0.5, "ending"], ["Uppsala", 0.5, "ending"], ["Visby", 0.3, "ending"]],
  capitalNoun: [["i", 48, "countryThenVerb"], ["heter", 28, "cityAnswer"], ["är", 20, "cityAnswer"], ["kallas", 3, "cityAnswer"], ["ligger", 1, "located"]],
  countryThenVerb: [["Sverige", 97, "capitalVerb"], ["landet", 1.5, "capitalVerb"], ["Norge", 0.5, "capitalVerb"], ["Danmark", 0.5, "capitalVerb"], ["Finland", 0.5, "capitalVerb"]],
  itCapital: [["är", 94, "itIsCapital"], ["blir", 3, "itIsCapital"], ["borde", 1, "beCapital"], ["kan", 1, "beCapital"], ["måste", 1, "beCapital"]],
  beCapital: [["vara", 94, "itIsCapital"], ["väl", 2, "beCapital"], ["ju", 2, "beCapital"], ["nog", 1, "beCapital"], ["faktiskt", 1, "beCapital"]],
  itIsCapital: [["Stockholm", 82, "stockholmTail"], ["Sveriges", 9, "capitalObject"], ["landets", 5, "capitalObject"], ["faktiskt", 3, "cityAnswer"], ["väl", 1, "cityAnswer"]],
  stockholmTail: [[".", 52, "sentenceStart"], ["i", 24, "inSweden"], [",", 13, "cityApposition"], ["!", 8, "sentenceStart"], ["som", 3, "relativeCity"]],
  iCapital: [["skulle", 58, "wouldSay"], ["tror", 20, "thatCapital"], ["säger", 10, "cityAnswer"], ["vet", 8, "thatCapital"], ["svarar", 4, "cityAnswer"]],
  wouldSay: [["säga", 90, "thatCapital"], ["svara", 5, "cityAnswer"], ["gissa", 3, "cityAnswer"], ["välja", 1, "cityAnswer"], ["tro", 1, "thatCapital"]],
  thatCapital: [["att", 86, "cityAnswer"], ["Stockholm", 9, "ending"], ["nog", 3, "thatCapital"], ["faktiskt", 1, "thatCapital"], ["väl", 1, "thatCapital"]],
  located: [["i", 83, "inSweden"], ["vid", 9, "waterside"], ["nära", 4, "waterside"], ["vackert", 3, "located"], ["öster", 1, "eastOf"]],
  inSweden: [["Sverige", 86, "ending"], ["Svealand", 7, "ending"], ["Skandinavien", 4, "ending"], ["Europa", 2, "ending"], ["Norden", 1, "ending"]],
  waterside: [["Mälaren", 37, "ending"], ["Östersjön", 28, "ending"], ["vattnet", 18, "ending"], ["kusten", 12, "ending"], ["skärgården", 5, "ending"]],
  eastOf: [["om", 92, "westPlace"], ["i", 4, "inSweden"], ["ut", 2, "ending"], ["längs", 1, "waterside"], ["vid", 1, "waterside"]],
  westPlace: [["Göteborg", 43, "ending"], ["Vänern", 22, "ending"], ["Örebro", 20, "ending"], ["Oslo", 10, "ending"], ["Karlstad", 5, "ending"]],
  cityApposition: [["Sveriges", 69, "capitalObject"], ["landets", 15, "capitalObject"], ["en", 10, "cityAdjective"], ["den", 5, "swedishAdjective"], ["ibland", 1, "called"]],
  called: [["kallad", 74, "cityNickname"], ["beskriven", 12, "asCity"], ["sedd", 7, "asCity"], ["omtalad", 5, "asCity"], ["benämnd", 2, "cityNickname"]],
  asCity: [["som", 90, "cityAdjective"], ["med", 4, "articleNoun"], ["i", 3, "inSweden"], ["av", 2, "people"], ["bland", 1, "people"]],
  cityNickname: [["Nordens", 70, "cityNoun"], ["huvudstaden", 15, "ending"], ["Venedig", 8, "ending"], ["storstan", 5, "ending"], ["Stockholm", 2, "ending"]],
  cityAdjective: [["stor", 49, "cityNoun"], ["svensk", 24, "cityNoun"], ["vacker", 15, "cityNoun"], ["nordisk", 8, "cityNoun"], ["livlig", 4, "cityNoun"]],
  swedishAdjective: [["svenska", 78, "capitalObject"], ["vackra", 8, "capitalObject"], ["stora", 7, "capitalObject"], ["nordiska", 5, "capitalObject"], ["livliga", 2, "capitalObject"]],
  cityNoun: [["stad", 77, "ending"], ["huvudstad", 12, "ending"], ["metropol", 6, "ending"], ["kommun", 3, "ending"], ["centralort", 2, "ending"]],
  politicalNoun: [["centrum", 70, "ending"], ["nav", 12, "ending"], ["hjärta", 10, "ending"], ["mittpunkt", 5, "ending"], ["mötesplats", 3, "ending"]],
  relativeCity: [["ligger", 48, "located"], ["är", 32, "isCapital"], ["har", 12, "cityHas"], ["kallas", 5, "cityNickname"], ["växer", 3, "ending"]],
  cityHas: [["många", 47, "cityFeatures"], ["flera", 25, "cityFeatures"], ["vackra", 15, "cityFeatures"], ["stora", 8, "cityFeatures"], ["gamla", 5, "cityFeatures"]],
  cityFeatures: [["öar", 45, "ending"], ["broar", 25, "ending"], ["byggnader", 15, "ending"], ["parker", 10, "ending"], ["museer", 5, "ending"]],

  sum: [["4", 60, "four"], ["Svaret", 18, "answer"], ["Det", 13, "itSum"], ["Summan", 8, "sumNoun"], ["Beräkningen", 1, "calculation"]],
  four: [["är", 69, "fourIs"], [".", 18, "mathSentence"], ["!", 8, "mathSentence"], [",", 4, "fourAside"], ["?", 1, "mathSentence"]],
  fourIs: [["det", 79, "answerAdjective"], ["ett", 12, "numberAdjective"], ["summan", 6, "sumNoun"], ["rätt", 2, "ending"], ["korrekt", 1, "ending"]],
  answerAdjective: [["rätta", 72, "answerNoun"], ["korrekta", 17, "answerNoun"], ["exakta", 6, "answerNoun"], ["enkla", 3, "answerNoun"], ["vanliga", 2, "answerNoun"]],
  answerNoun: [["svaret", 79, "ending"], ["resultatet", 15, "ending"], ["talet", 3, "ending"], ["värdet", 2, "ending"], ["antalet", 1, "ending"]],
  numberAdjective: [["jämnt", 65, "numberNoun"], ["helt", 15, "numberNoun"], ["positivt", 12, "numberNoun"], ["litet", 5, "numberNoun"], ["naturligt", 3, "numberNoun"]],
  numberNoun: [["tal", 96, "ending"], ["heltal", 1, "ending"], ["resultat", 1, "ending"], ["värde", 1, "ending"], ["antal", 1, "ending"]],
  fourAside: [["det", 62, "itSum"], ["eftersom", 18, "two"], ["alltså", 11, "ending"], ["precis", 6, "ending"], ["ja", 3, "ending"]],
  answer: [["på", 52, "theQuestion"], ["är", 41, "numberAnswer"], ["blir", 5, "numberAnswer"], ["ska", 1, "beNumber"], ["borde", 1, "beNumber"]],
  theQuestion: [["frågan", 85, "numberVerb"], ["uppgiften", 9, "numberVerb"], ["uträkningen", 3, "numberVerb"], ["addition", 2, "numberVerb"], ["detta", 1, "numberVerb"]],
  itSum: [["rätta", 70, "answerThenVerb"], ["korrekta", 16, "answerThenVerb"], ["blir", 7, "numberAnswer"], ["är", 5, "numberAnswer"], ["enkla", 2, "answerThenVerb"]],
  answerThenVerb: [["svaret", 81, "numberVerb"], ["resultatet", 13, "numberVerb"], ["talet", 3, "numberVerb"], ["värdet", 2, "numberVerb"], ["antalet", 1, "numberVerb"]],
  numberVerb: [["är", 83, "numberAnswer"], ["blir", 12, "numberAnswer"], ["motsvarar", 3, "numberAnswer"], ["ger", 1, "numberAnswer"], ["förblir", 1, "numberAnswer"]],
  numberAnswer: [["4", 98, "ending"], ["fyra", 1.4, "ending"], ["5", 0.25, "ending"], ["3", 0.2, "ending"], ["22", 0.15, "ending"]],
  beNumber: [["vara", 92, "numberAnswer"], ["bli", 5, "numberAnswer"], ["nog", 1, "beNumber"], ["väl", 1, "beNumber"], ["alltid", 1, "beNumber"]],
  sumNoun: [["av", 78, "theNumbers"], ["är", 16, "numberAnswer"], ["blir", 4, "numberAnswer"], ["ska", 1, "beNumber"], ["borde", 1, "beNumber"]],
  theNumbers: [["talen", 65, "numberVerb"], ["tvåorna", 20, "numberVerb"], ["dessa", 8, "numbersThenVerb"], ["två", 5, "andTwo"], ["termerna", 2, "numberVerb"]],
  numbersThenVerb: [["tal", 69, "numberVerb"], ["tvåor", 15, "numberVerb"], ["termer", 8, "numberVerb"], ["värden", 5, "numberVerb"], ["heltal", 3, "numberVerb"]],
  andTwo: [["och", 88, "two"], ["plus", 8, "two"], ["tal", 2, "numberVerb"], ["tvåor", 1, "numberVerb"], ["heltal", 1, "numberVerb"]],
  two: [["2", 60, "numberVerb"], ["två", 32, "numberVerb"], ["tvåor", 4, "numberVerb"], ["ett", 2, "numberVerb"], ["tre", 2, "numberVerb"]],
  calculation: [["ger", 69, "calculationGives"], ["visar", 19, "calculationGives"], ["blir", 8, "numberAnswer"], ["är", 3, "numberAnswer"], ["resulterar", 1, "resultIn"]],
  calculationGives: [["svaret", 64, "numberAnswer"], ["resultatet", 22, "numberAnswer"], ["4", 10, "ending"], ["fyra", 3, "ending"], ["alltså", 1, "numberAnswer"]],
  resultIn: [["i", 95, "numberAnswer"], ["alltså", 2, "resultIn"], ["här", 1, "resultIn"], ["också", 1, "resultIn"], ["oftast", 1, "resultIn"]],
  mathSentence: [["Det", 48, "itSum"], ["Två", 23, "andTwo"], ["Svaret", 15, "answer"], ["Summan", 10, "sumNoun"], ["Beräkningen", 4, "calculation"]],

  story: [["hittade", 39, "found"], ["såg", 30, "saw"], ["möttes", 14, "met"], ["klev", 11, "stepped"], ["hörde", 6, "heard"]],
  found: [["en", 57, "storyNoun"], ["ett", 29, "storyNeuter"], ["ingenting", 7, "nothingTail"], ["sin", 5, "familiarNoun"], ["plötsligt", 2, "found"]],
  saw: [["en", 53, "storyNoun"], ["ett", 28, "storyNeuter"], ["sin", 9, "familiarNoun"], ["ingenting", 6, "nothingTail"], ["något", 4, "something"]],
  storyNoun: [["katt", 32, "catTail"], ["människa", 26, "personTail"], ["trädgård", 20, "gardenTail"], ["robot", 14, "robotTail"], ["drake", 8, "dragonTail"]],
  storyNeuter: [["rum", 33, "roomTail"], ["brev", 25, "letterTail"], ["ljus", 18, "lightTail"], ["barn", 14, "personTail"], ["monster", 10, "dragonTail"]],
  familiarNoun: [["vän", 46, "personTail"], ["skapare", 26, "personTail"], ["katt", 14, "catTail"], ["granne", 9, "personTail"], ["familj", 5, "personTail"]],
  catTail: [["som", 70, "catAction"], ["i", 11, "place"], ["med", 9, "catFeature"], [".", 7, "storySentence"], ["!", 3, "storySentence"]],
  personTail: [["som", 73, "personAction"], ["med", 12, "personFeature"], ["i", 8, "place"], [".", 5, "storySentence"], ["!", 2, "storySentence"]],
  gardenTail: [["full", 40, "gardenFullOf"], ["med", 27, "gardenFeature"], ["som", 17, "gardenAction"], ["i", 10, "place"], [".", 6, "storySentence"]],
  robotTail: [["som", 68, "robotAction"], ["med", 17, "robotFeature"], ["i", 8, "place"], [".", 5, "storySentence"], ["!", 2, "storySentence"]],
  dragonTail: [["som", 67, "dragonAction"], ["med", 18, "dragonFeature"], ["i", 7, "place"], [".", 5, "storySentence"], ["!", 3, "storySentence"]],
  roomTail: [["fullt", 43, "fullOf"], ["med", 27, "roomFeature"], ["som", 18, "roomAction"], ["i", 7, "place"], [".", 5, "storySentence"]],
  letterTail: [["som", 51, "letterAction"], ["från", 25, "sender"], ["med", 16, "letterFeature"], [".", 6, "storySentence"], ["!", 2, "storySentence"]],
  lightTail: [["som", 55, "lightAction"], ["i", 19, "place"], ["från", 14, "lightSource"], [".", 9, "storySentence"], ["!", 3, "storySentence"]],
  catAction: [["sov", 39, "ending"], ["spann", 25, "ending"], ["jamade", 19, "ending"], ["väntade", 12, "ending"], ["talade", 5, "ending"]],
  personAction: [["log", 36, "ending"], ["väntade", 29, "ending"], ["vinkade", 17, "ending"], ["sov", 11, "ending"], ["grät", 7, "ending"]],
  gardenAction: [["blomstrade", 42, "ending"], ["glittrade", 24, "ending"], ["doftade", 16, "ending"], ["växte", 12, "ending"], ["viskade", 6, "ending"]],
  robotAction: [["väntade", 33, "ending"], ["vinkade", 26, "ending"], ["log", 20, "ending"], ["dansade", 14, "ending"], ["sov", 7, "ending"]],
  dragonAction: [["sov", 32, "ending"], ["log", 25, "ending"], ["mullrade", 19, "ending"], ["dansade", 14, "ending"], ["nös", 10, "ending"]],
  roomAction: [["glittrade", 36, "ending"], ["ekade", 25, "ending"], ["snurrade", 17, "ending"], ["lyste", 14, "ending"], ["svävade", 8, "ending"]],
  letterAction: [["glödde", 28, "ending"], ["väntade", 24, "ending"], ["fladdrade", 22, "ending"], ["viskade", 16, "ending"], ["sjöng", 10, "ending"]],
  lightAction: [["blinkade", 36, "ending"], ["glittrade", 28, "ending"], ["fladdrade", 18, "ending"], ["dansade", 12, "ending"], ["slocknade", 6, "ending"]],
  catFeature: [["gröna", 33, "eyes"], ["stora", 26, "eyes"], ["svart", 19, "fur"], ["mjuk", 15, "fur"], ["vita", 7, "paws"]],
  personFeature: [["ett", 45, "smile"], ["stora", 23, "eyes"], ["gröna", 15, "eyes"], ["en", 10, "storyNoun"], ["öppna", 7, "arms"]],
  robotFeature: [["blå", 36, "eyes"], ["röda", 24, "eyes"], ["gröna", 18, "eyes"], ["stora", 14, "eyes"], ["fyra", 8, "arms"]],
  dragonFeature: [["gyllene", 34, "wings"], ["stora", 27, "wings"], ["gröna", 19, "eyes"], ["röda", 12, "eyes"], ["blå", 8, "wings"]],
  gardenFeature: [["blommor", 47, "ending"], ["rosor", 22, "ending"], ["fjärilar", 16, "ending"], ["träd", 10, "ending"], ["robotar", 5, "ending"]],
  roomFeature: [["böcker", 38, "ending"], ["maskiner", 24, "ending"], ["lampor", 18, "ending"], ["leksaker", 13, "ending"], ["stjärnor", 7, "ending"]],
  letterFeature: [["hemligheter", 40, "ending"], ["instruktioner", 25, "ending"], ["bilder", 17, "ending"], ["ritningar", 11, "ending"], ["önskningar", 7, "ending"]],
  fullOf: [["av", 94, "roomFeature"], ["med", 3, "gardenFeature"], ["och", 1, "lightAction"], [".", 1, "storySentence"], [",", 1, "storySentence"]],
  gardenFullOf: [["av", 94, "gardenFeature"], ["med", 3, "gardenFeature"], ["och", 1, "gardenAction"], [".", 1, "storySentence"], [",", 1, "storySentence"]],
  eyes: [["ögon", 90, "ending"], ["ögonlinser", 4, "ending"], ["lampor", 3, "ending"], ["sensorer", 2, "ending"], ["glasögon", 1, "ending"]],
  fur: [["päls", 90, "ending"], ["svans", 4, "ending"], ["nos", 3, "ending"], ["krage", 2, "ending"], ["mössa", 1, "ending"]],
  paws: [["tassar", 88, "ending"], ["fläckar", 6, "ending"], ["öron", 3, "ending"], ["morrhår", 2, "ending"], ["vantar", 1, "ending"]],
  wings: [["vingar", 84, "ending"], ["fjäll", 8, "ending"], ["horn", 4, "ending"], ["klor", 3, "ending"], ["ögon", 1, "ending"]],
  arms: [["armar", 78, "ending"], ["händer", 11, "ending"], ["vingar", 5, "ending"], ["dörrar", 4, "ending"], ["fönster", 2, "ending"]],
  smile: [["leende", 75, "ending"], ["brev", 10, "ending"], ["paket", 7, "ending"], ["paraply", 5, "ending"], ["svärd", 3, "ending"]],
  sender: [["skaparen", 36, "ending"], ["framtiden", 29, "ending"], ["jorden", 18, "ending"], ["rymden", 11, "ending"], ["månen", 6, "ending"]],
  lightSource: [["solen", 45, "ending"], ["månen", 22, "ending"], ["lampan", 16, "ending"], ["fönstret", 11, "ending"], ["stjärnorna", 6, "ending"]],
  nothingTail: [["annat", 33, "than"], ["alls", 29, "ending"], [".", 22, "storySentence"], ["förutom", 12, "articleNoun"], ["!", 4, "storySentence"]],
  than: [["än", 94, "articleNoun"], [".", 2, "storySentence"], [",", 2, "storySentence"], ["där", 1, "ending"], ["heller", 1, "ending"]],
  something: [["märkligt", 39, "ending"], ["oväntat", 27, "ending"], ["vackert", 18, "ending"], ["skrämmande", 10, "ending"], ["magiskt", 6, "ending"]],
  met: [["av", 94, "articleNoun"], ["plötsligt", 3, "met"], ["där", 1.5, "met"], ["genast", 1, "met"], ["sedan", 0.5, "met"]],
  articleNoun: [["en", 51, "storyNoun"], ["ett", 29, "storyNeuter"], ["sin", 10, "familiarNoun"], ["något", 7, "something"], ["ingenting", 3, "nothingTail"]],
  stepped: [["ut", 59, "stepPlace"], ["in", 28, "stepPlace"], ["försiktigt", 7, "stepped"], ["tyst", 4, "stepped"], ["rakt", 2, "stepped"]],
  stepPlace: [["i", 54, "place"], ["på", 19, "surface"], ["genom", 14, "passage"], ["under", 8, "shelter"], ["bland", 5, "crowd"]],
  place: [["trädgården", 32, "placeTail"], ["rummet", 27, "placeTail"], ["mörkret", 21, "placeTail"], ["solskenet", 13, "placeTail"], ["rymden", 7, "placeTail"]],
  surface: [["gatan", 38, "placeTail"], ["trappan", 28, "placeTail"], ["gården", 18, "placeTail"], ["balkongen", 11, "placeTail"], ["månen", 5, "placeTail"]],
  passage: [["porten", 39, "placeTail"], ["dörren", 30, "placeTail"], ["dimman", 16, "placeTail"], ["tunneln", 10, "placeTail"], ["spegeln", 5, "placeTail"]],
  shelter: [["trädet", 43, "placeTail"], ["taket", 24, "placeTail"], ["stjärnorna", 17, "placeTail"], ["bron", 11, "placeTail"], ["paraplyet", 5, "placeTail"]],
  crowd: [["människorna", 37, "placeTail"], ["träden", 28, "placeTail"], ["blommorna", 18, "placeTail"], ["robotarna", 11, "placeTail"], ["stjärnorna", 6, "placeTail"]],
  placeTail: [[".", 61, "storySentence"], ["och", 18, "story"], [",", 11, "story"], ["!", 7, "storySentence"], ["där", 3, "story"]],
  heard: [["en", 40, "soundNoun"], ["ett", 27, "soundNeuter"], ["musik", 17, "soundFrom"], ["någon", 10, "someoneSound"], ["ingenting", 6, "nothingTail"]],
  soundNoun: [["röst", 42, "soundTail"], ["sång", 26, "soundTail"], ["smäll", 17, "soundTail"], ["viskning", 10, "soundTail"], ["melodi", 5, "soundTail"]],
  soundNeuter: [["ljud", 41, "soundTail"], ["skratt", 27, "soundTail"], ["rop", 17, "soundTail"], ["muller", 10, "soundTail"], ["knackande", 5, "soundTail"]],
  soundTail: [["från", 44, "soundLocation"], ["som", 31, "soundAction"], ["i", 14, "place"], [".", 8, "storySentence"], ["!", 3, "storySentence"]],
  soundFrom: [["från", 68, "soundLocation"], ["i", 18, "place"], ["som", 8, "soundAction"], [".", 4, "storySentence"], ["!", 2, "storySentence"]],
  soundLocation: [["rummet", 37, "ending"], ["trädgården", 27, "ending"], ["skogen", 19, "ending"], ["källaren", 11, "ending"], ["rymden", 6, "ending"]],
  soundAction: [["ekade", 39, "ending"], ["viskade", 25, "ending"], ["lockade", 18, "ending"], ["tystnade", 12, "ending"], ["skrämde", 6, "ending"]],
  someoneSound: [["som", 61, "someoneAction"], ["sjunga", 19, "ending"], ["skratta", 11, "ending"], ["ropa", 6, "ending"], ["viska", 3, "ending"]],
  someoneAction: [["sjöng", 34, "ending"], ["skrattade", 27, "ending"], ["ropade", 20, "ending"], ["viskade", 13, "ending"], ["grät", 6, "ending"]],
  storySentence: [["Roboten", 51, "story"], ["Den", 22, "story"], ["Där", 13, "story"], ["Plötsligt", 9, "story"], ["Sedan", 5, "story"]],

  ending: [[".", 88, "sentenceStart"], [",", 5, "continuation"], ["!", 4, "sentenceStart"], ["också", 2, "ending"], ["?", 1, "sentenceStart"]],
  continuation: [["och", 49, "sentenceStart"], ["som", 24, "relativeCity"], ["men", 13, "sentenceStart"], ["vilket", 9, "capitalVerb"], ["eller", 5, "cityAnswer"]],
  sentenceStart: [["Det", 43, "itCapital"], ["Staden", 25, "citySentenceVerb"], ["Den", 17, "citySentenceVerb"], ["Stockholm", 10, "stockholm"], ["Här", 5, "citySentenceVerb"]],
  citySentenceVerb: [["är", 51, "isCapital"], ["ligger", 27, "located"], ["har", 14, "cityHas"], ["kallas", 5, "cityNickname"], ["växer", 3, "ending"]],
  people: [["många", 45, "peopleNoun"], ["svenskar", 25, "ending"], ["turister", 15, "ending"], ["besökare", 10, "ending"], ["invånare", 5, "ending"]],
  peopleNoun: [["svenskar", 38, "ending"], ["turister", 25, "ending"], ["besökare", 18, "ending"], ["invånare", 12, "ending"], ["människor", 7, "ending"]]
};

interface PrefixNode { candidates: readonly Candidate[]; children: ReadonlyMap<string, PrefixNode> }
const databases = new Map<PromptId, PrefixNode>();

function expand(ruleId: string, depth: number, promptId: PromptId): PrefixNode {
  if (depth === ANSWER_LENGTH) return { candidates: [], children: new Map() };
  // Sentence boundaries retain the original task: arithmetic must not suddenly
  // continue with vocabulary about Stockholm, nor a story with a geography answer.
  const contextRule = ruleId === "sentenceStart" && promptId !== "capital"
    ? promptId === "sum" ? "mathSentence" : "storySentence"
    : ruleId;
  const rule = rules[contextRule];
  if (!rule || rule.length !== 5 || new Set(rule.map(([token]) => token)).size !== 5) {
    throw new Error(`Ogiltig databasregel: ${ruleId}`);
  }
  const total = rule.reduce((sum, [, weight]) => sum + weight, 0);
  return {
    candidates: rule.map(([token, weight]) => ({ token, probability: weight / total })),
    children: new Map(rule.map(([token, , next]) => [token, expand(next, depth + 1, promptId)]))
  };
}

/** Materialize all 781 input contexts for a prompt, then use exact-prefix lookup. */
export function getBaseCandidates(promptId: PromptId, tokens: readonly string[]): readonly Candidate[] {
  const prompt = PROMPTS.find(({ id }) => id === promptId);
  if (!prompt) throw new RangeError("Okänd prompt.");
  if (tokens.length >= ANSWER_LENGTH) throw new RangeError("Svaret är redan färdigt.");
  let node = databases.get(promptId);
  if (!node) {
    node = expand(prompt.start, 0, promptId);
    databases.set(promptId, node);
  }
  for (const token of tokens) {
    const child: PrefixNode | undefined = node.children.get(token);
    if (!child) throw new RangeError("Den här texten saknas i databasen.");
    node = child;
  }
  return node.candidates;
}
