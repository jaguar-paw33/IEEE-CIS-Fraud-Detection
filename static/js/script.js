function makeMergedVisible() {
  $("#form-merged")
    .css({
      opacity: "0",
      display: "flex",
    })
    .show()
    .animate({ opacity: 1 }, 100);
  document.getElementById("form-non-merged").style.display = "none";
}

function makeNonMergedVisible() {
  $("#form-non-merged")
    .css({
      opacity: "0",
      display: "flex",
    })
    .show()
    .animate({ opacity: 1 }, 100);
  document.getElementById("form-merged").style.display = "none";
}
