// Uplink decoder in the The Things Stack format (decodeUplink), twin of src/payload.py.
// Byte 0: version, byte 1: status, then 12 big-endian int16 = round(log10(feature) * 1000).
var NAMES = ["rms", "peak", "crest", "kurtosis", "band0", "band1", "band2", "band3", "band4", "band5", "band6", "band7"];

function decodeUplink(input) {
  var b = input.bytes;
  if (b.length < 2 || b[0] !== 1) return { errors: ["unknown frame format"] };
  var data = { status: b[1] ? "alert" : "normal" };
  if (b.length === 2) return { data: data };
  if (b.length !== 2 + 2 * NAMES.length) return { errors: ["feature frame of " + b.length + " bytes"] };
  data.features = {};
  for (var i = 0; i < NAMES.length; i++) {
    var v = (b[2 + 2 * i] << 8) | b[3 + 2 * i];
    if (v & 0x8000) v -= 0x10000;
    data.features[NAMES[i]] = Math.pow(10, v / 1000);
  }
  return { data: data };
}

if (typeof module !== "undefined") module.exports = { decodeUplink: decodeUplink };
