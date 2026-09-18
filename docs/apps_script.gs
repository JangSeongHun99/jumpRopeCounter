/**
 * 줄넘기 카운터 - 구글 스프레드시트 연동 (Apps Script 웹 앱)
 *
 * 설치 (선생님이 한 번만)
 *   1. 구글 드라이브에서 새 스프레드시트를 만듭니다 (예: "줄넘기 기록").
 *   2. 메뉴 확장 프로그램 > Apps Script 를 열고, 기본 코드를 지운 뒤 이 파일 내용을 전부 붙여넣고 저장합니다.
 *   3. 오른쪽 위 배포 > 새 배포 > 유형 선택(톱니바퀴) > 웹 앱
 *        - 설명: 줄넘기
 *        - 다음 사용자 인증 정보로 실행: 나
 *        - 액세스 권한이 있는 사용자: 모든 사용자   <- 꼭 "모든 사용자"
 *      배포 를 누르고 권한 허용(고급 > 안전하지 않은 페이지로 이동 > 허용)을 거치면
 *      https://script.google.com/macros/s/..../exec 형태의 웹 앱 URL이 나옵니다.
 *   4. 그 URL을 줄넘기 카운터 페이지의 "선생님 설정 > 구글 스프레드시트 연동"에 넣고 "연결 테스트"를 누릅니다.
 *      "학생용 링크 복사"로 만든 링크를 학생들에게 나눠 주면 학생 아이패드에는 설정이 필요 없습니다.
 *
 * 동작: 저장할 때마다 학년-반 이름의 탭(없으면 만들어짐)과 "전체" 탭에 한 줄씩 추가됩니다.
 *       코드를 고친 뒤에는 배포 > 배포 관리 > 새 버전 으로 다시 배포해야 반영됩니다.
 */

var ALL_SHEET = "전체";
var HEADER = ["기록 시각", "학년-반", "이름", "점프 횟수", "시간(초)", "분당 횟수", "걸림", "최고 연속", "제한 시간(초)", "서버 수신 시각", "기기"];

function doPost(e) {
  var lock = LockService.getScriptLock();
  lock.waitLock(20000);   // 여러 아이패드가 동시에 보내도 순서대로 처리
  try {
    var d = JSON.parse(e.postData.contents);
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    if (d.type === "ping") return json_({ ok: true, spreadsheet: ss.getName() });
    if (d.type !== "record") return json_({ ok: false, error: "unknown type" });
    var cls = String(d.cls || "").trim().replace(/[\[\]\*\/\\\?:]/g, "-").slice(0, 60) || "미지정";
    var row = [String(d.date || ""), cls, String(d.name || ""), Number(d.count) || 0, Number(d.duration) || 0,
               Number(d.perMin) || 0, Number(d.misses) || 0, Number(d.best) || 0, Number(d.timer) || 0,
               new Date(), String(d.device || "")];
    appendTo_(ss, cls, row);
    appendTo_(ss, ALL_SHEET, row);
    return json_({ ok: true, sheet: cls });
  } catch (err) {
    return json_({ ok: false, error: String(err) });
  } finally {
    lock.releaseLock();
  }
}

function doGet() {
  return json_({ ok: true, message: "줄넘기 카운터 시트 연동이 동작 중입니다. 페이지의 연결 테스트를 사용하세요." });
}

function appendTo_(ss, name, row) {
  var sh = ss.getSheetByName(name);
  if (!sh) {
    sh = ss.insertSheet(name);
    sh.appendRow(HEADER);
    sh.getRange(1, 1, 1, HEADER.length).setFontWeight("bold");
    sh.setFrozenRows(1);
  } else if (sh.getLastRow() === 0) {
    sh.appendRow(HEADER);
    sh.setFrozenRows(1);
  }
  sh.appendRow(row);
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}
