function logData() {
  var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = spreadsheet.getSheetByName("Sheet1"); // your sheet name 

  // 1) Get your input data block (DISPLAY text)
  var dataRange = sheet.getRange("C11:I11");     // input area
  var data = dataRange.getDisplayValues();       // 2D array (1 row x 7 cols)

  // 2) Figure out where the log should start/append
  var LOG_START_ROW = 23;   // first row of your log block
  var START_COLUMN = 3;     // column C = 3

  var maxRow = sheet.getLastRow();   // highest row with *anything* on the sheet
  var firstEmptyRow = LOG_START_ROW; // default if no log data yet

  if (maxRow >= LOG_START_ROW) {
    // Look ONLY at columns C–I from LOG_START_ROW down to maxRow
    var height = maxRow - LOG_START_ROW + 1;
    var logRange = sheet.getRange(LOG_START_ROW, START_COLUMN, height, data[0].length); // C23:I(maxRow)
    var logValues = logRange.getValues();

    // Scan from bottom up to find the last row that has ANY data in C–I
    for (var r = logValues.length - 1; r >= 0; r--) {
      var rowHasData = false;
      for (var c = 0; c < logValues[0].length; c++) {
        if (logValues[r][c] !== "" && logValues[r][c] !== null) {
          rowHasData = true;
          break;
        }
      }
      if (rowHasData) {
        firstEmptyRow = LOG_START_ROW + r + 1; // row after the last used one
        break;
      }
    }
    // If we never found a row with data, firstEmptyRow stays LOG_START_ROW
  }

  // 3) Write the new log row at the first empty row in C–I
  sheet
    .getRange(firstEmptyRow, START_COLUMN, data.length, data[0].length)
    .setValues(data);

  // 4) Clear the input cells
  dataRange.clearContent();

  // 5) Reset checkboxes in a 2D box range
  var checkboxRange = sheet.getRange("K5:U14");
  var checkboxValues = checkboxRange.getValues(); // 2D array

  for (var r = 0; r < checkboxValues.length; r++) {
    for (var c = 0; c < checkboxValues[r].length; c++) {
      if (checkboxValues[r][c] === true) {
        // Only uncheck if currently checked
        checkboxValues[r][c] = false;
      }
      // If it's false or empty, we leave it as-is
    }
  }

  // Write updated checkbox values back
  checkboxRange.setValues(checkboxValues);

  // Popup message
  SpreadsheetApp.getActive().toast("Data logged ✅");
}
