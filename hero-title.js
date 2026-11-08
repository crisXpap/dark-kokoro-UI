/* Hero title: multi-axis variable-font specimen morph.
   Every character (spaces and punctuation included) gets its own
   font-variation-settings, recomputed from its position in the full title:

     · wght  900 → 100   bold to thin, linear drop across the whole string
     · wdth  100 → 145 → 75   standard, stretching to a wide peak near the
                         start-middle, then condensing toward the end
     · slnt -10 →  0     heavy slant on the very first character, smoothly
                         straightening out across the first third

   Formula, for index i of N total characters:
     progress = i / (N - 1)

   Special case: the word "Pebble" reuses the exact weight values the line
   below it ("your local AI voice.") computes, reversed — light on the
   left, progressively bolder toward the right of the word.

   Second special case: "TTS" starts exactly as bold as Pebble's last
   letter ("e") and keeps getting lighter with each character, at the
   same per-character rate every other word in the title steps down by. */
(function () {
  'use strict';

  var LINES = ['Pebble TTS', 'your local AI voice.'];
  var MIRROR_WORD = 'Pebble';

  var WGHT_MAX = 900;
  var WGHT_MIN = 100;
  var WDTH_BASE = 100;
  var WDTH_PEAK = 145; // stretched middle
  var WDTH_MIN = 75;   // condensed end
  var SLNT_START = -10; // Roboto Flex slnt axis runs -10 to 0

  var title = document.getElementById('hero-title');
  if (!title) return;

  // The whole title counts as one continuous string; the '\n' between the
  // lines stays in N so the morph flows across both lines.
  var fullText = LINES.join('\n');
  var total = fullText.length;

  function variationFor(i) {
    var progress = total > 1 ? i / (total - 1) : 0;

    // 1. Weight: 900 down to 100
    var wght = Math.round(WGHT_MAX - progress * (WGHT_MAX - WGHT_MIN));

    // 2. Slant: -10deg on character 0, tapering quickly to 0deg
    var slnt = i === 0
      ? SLNT_START
      : Math.min(0, Math.round(SLNT_START + progress * 30));

    // 3. Width: standard, peak wide around 30% progress, end condensed
    var wdth;
    if (progress < 0.3) {
      wdth = Math.round(WDTH_BASE + (progress / 0.3) * (WDTH_PEAK - WDTH_BASE));
    } else {
      wdth = Math.round(WDTH_PEAK - ((progress - 0.3) / 0.7) * (WDTH_PEAK - WDTH_MIN));
    }

    return { wght: wght, slnt: slnt, wdth: wdth };
  }

  // The weights the bottom line ("right on your device.") will actually
  // use, in render order — "MIRROR_WORD" borrows them reversed.
  var belowStart = total - LINES[1].length;
  var belowWeights = [];
  for (var k = belowStart; k < total; k++) {
    belowWeights.push(variationFor(k).wght);
  }

  // Char j of the word maps onto the bottom line's weight list, mirrored:
  // word start samples the list's end (lightest), word end its start (boldest).
  function mirroredWeight(j) {
    var pos = Math.round(
      (1 - j / (MIRROR_WORD.length - 1)) * (belowWeights.length - 1)
    );
    return belowWeights[pos];
  }

  // "TTS" header word: as bold as Pebble's last letter at its first "T",
  // then descending per character at the title-wide step (weight units
  // lost per character along the global 900 → 100 ramp).
  var wordEndWeight = mirroredWeight(MIRROR_WORD.length - 1);
  var perCharStep = (WGHT_MAX - WGHT_MIN) / (total - 1);

  var i = 0;
  LINES.forEach(function (lineText, lineIndex) {
    var lineEl = title.querySelector('.hero-line[data-line="' + lineIndex + '"]');
    if (!lineEl) return;

    if (lineIndex > 0) i++; // skip the '\n' separator — it still counts in N
    lineEl.textContent = '';

    Array.prototype.forEach.call(lineText, function (ch, chIndex) {
      var v = variationFor(i);
      if (lineIndex === 0 && chIndex < MIRROR_WORD.length) {
        v.wght = mirroredWeight(chIndex);
      } else if (lineIndex === 0 && chIndex > MIRROR_WORD.length) {
        // First "T" (j = 0) is exactly as bold as Pebble's last "e"; each
        // following character drops by the title-wide rate. The space
        // between the words keeps its untouched global value.
        var j = chIndex - MIRROR_WORD.length - 1;
        v.wght = Math.round(wordEndWeight - j * perCharStep);
      }

      // Each axis tag must be quoted individually for a valid value.
      var settings = "'wght' " + v.wght + ", 'wdth' " + v.wdth + ", 'slnt' " + v.slnt;

      var span = document.createElement('span');
      span.className = 'hero-char';
      span.textContent = ch;
      span.style.fontVariationSettings = settings;

      lineEl.appendChild(span);
      i++;
    });

    // The sticker defined in index.html sits as a direct <h1> child so the
    // textContent wipe above can't remove it; move it inline after the first
    // line's characters so it rides beside the last glyph of the main line.
    if (lineIndex === 0) {
      var sticker = document.getElementById('hero-sticker');
      if (sticker) lineEl.appendChild(sticker);
    }
  });
})();
