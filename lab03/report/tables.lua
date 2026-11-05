-- Допоміжний модуль Lua для завантаження та форматування даних CSV у LaTeX-таблиці
local RES = "../results/"

local function read_csv(name)
  local rows = {}
  local path = RES .. name
  local f = io.open(path, "r")
  if not f then
    f = io.open("results/" .. name, "r")
  end
  if not f then
    tex.print("\\textbf{ПОМИЛКА: " .. name .. " не знайдено}")
    return {}
  end
  for line in f:lines() do
    local r = {}
    for cell in (line .. ","):gmatch("([^,]*),") do
      r[#r + 1] = cell
    end
    rows[#rows + 1] = r
  end
  f:close()
  if #rows > 0 then
    table.remove(rows, 1) -- видаляємо рядок заголовків
  end
  return rows
end

local function out(cells)
  tex.sprint(table.concat(cells, " & ") .. " \\\\")
end

local function f(x, d)
  local v = tonumber(x)
  if not v then return tostring(x) end
  return string.format("%." .. (d or 2) .. "f", v)
end

-- Таблиця 1: Звірка власної реалізації Гарріса з cv2.cornerHarris (Частина A)
function tab_harris_verification()
  local rows = read_csv("a_harris_verification.csv")
  for _, r in ipairs(rows) do
    out({
      "\\texttt{" .. r[1] .. "}",
      r[2],
      f(r[3], 1) .. "\\%",
      f(r[4], 1) .. "\\%",
      f(r[5], 2) .. "~пк",
      f(r[6], 1) .. "\\%",
      f(r[7], 1) .. "\\%",
      f(r[8], 1) .. "\\%"
    })
  end
end

-- Таблиця 2: Чутливість детектора Гарріса до масштабу (Частина B)
function tab_scale_sensitivity()
  local rows = read_csv("b_scale_sensitivity.csv")
  for _, r in ipairs(rows) do
    out({
      "\\texttt{" .. r[1] .. "}",
      r[2],
      r[3],
      f(r[4], 1) .. "\\%",
      f(r[5], 1) .. "\\%",
      f(r[6], 1) .. "\\%",
      f(r[7], 2) .. "~пк",
      f(r[8], 1) .. "\\%"
    })
  end
end

-- Таблиця 3: Порівняльні характеристики SIFT та ORB (Частина C)
function tab_detector_benchmarks()
  local rows = read_csv("c_detector_benchmarks.csv")
  for _, r in ipairs(rows) do
    out({
      "\\texttt{" .. r[1] .. "}",
      r[2],
      f(r[3], 1) .. "~мс",
      r[4],
      f(r[7], 1) .. "~мс",
      r[8],
      f(r[11], 1) .. "~мс",
      r[12]
    })
  end
end

-- Таблиця 4: Повторюваність за повороту кадру (Частина D)
function tab_rep_rotation()
  local rows = read_csv("d_repeatability_rotation.csv")
  for _, r in ipairs(rows) do
    out({
      r[1] .. "$^\\circ$",
      f(r[2], 1) .. " $\\pm$ " .. f(r[3], 1),
      f(r[4], 1) .. " $\\pm$ " .. f(r[5], 1),
      f(r[6], 1) .. " $\\pm$ " .. f(r[7], 1)
    })
  end
end

-- Таблиця 5: Повторюваність за зміни масштабу (Частина D)
function tab_rep_scale()
  local rows = read_csv("d_repeatability_scale.csv")
  for _, r in ipairs(rows) do
    out({
      r[1],
      f(r[2], 1) .. " $\\pm$ " .. f(r[3], 1),
      f(r[4], 1) .. " $\\pm$ " .. f(r[5], 1),
      f(r[6], 1) .. " $\\pm$ " .. f(r[7], 1)
    })
  end
end

-- Таблиця 6: Повторюваність за зміни яскравості (Частина D)
function tab_rep_brightness()
  local rows = read_csv("d_repeatability_brightness.csv")
  for _, r in ipairs(rows) do
    out({
      r[1],
      f(r[2], 1) .. " $\\pm$ " .. f(r[3], 1),
      f(r[4], 1) .. " $\\pm$ " .. f(r[5], 1),
      f(r[6], 1) .. " $\\pm$ " .. f(r[7], 1)
    })
  end
end

-- Таблиця 7: Повторюваність за наявності шуму (Частина D)
function tab_rep_noise()
  local rows = read_csv("d_repeatability_noise.csv")
  for _, r in ipairs(rows) do
    out({
      r[1],
      f(r[2], 1) .. " $\\pm$ " .. f(r[3], 1),
      f(r[4], 1) .. " $\\pm$ " .. f(r[5], 1),
      f(r[6], 1) .. " $\\pm$ " .. f(r[7], 1)
    })
  end
end

