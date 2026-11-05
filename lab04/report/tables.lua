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
    line = line:gsub("\r", "")
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

local function esc_tex(s)
  return tostring(s):gsub("_", "\\_"):gsub("%%", "\\%%"):gsub("<", "$<$"):gsub(">", "$>$")
end

-- Таблиця 1: Критерій відношення відстаней Лоу та взаємна перевірка (Частина A)
function tab_ratio_test(filter_pair)
  local rows = read_csv("a_ratio_test.csv")
  for _, r in ipairs(rows) do
    if not filter_pair or r[1] == filter_pair then
      out({
        "\\texttt{" .. esc_tex(r[1]) .. "}",
        r[5],
        r[6],
        f(r[7], 2) .. "\\%",
        r[8],
        f(r[9], 1) .. "\\%"
      })
    end
  end
end

-- Таблиця 2: Порівняння класичного МНК та RANSAC (Частина B)
function tab_ols_vs_ransac()
  local rows = read_csv("b_ols_vs_ransac.csv")
  for _, r in ipairs(rows) do
    out({
      "\\texttt{" .. esc_tex(r[1]) .. "}",
      r[2],
      r[3] .. " (" .. f(100 - tonumber(r[5]), 1) .. "\\%)",
      r[4] .. " (" .. f(r[5], 1) .. "\\%)",
      f(r[6], 1) .. "~пк",
      f(r[9], 1) .. "~пк",
      f(r[11], 2) .. "~пк",
      f(r[12], 2) .. "~пк"
    })
  end
end

-- Таблиця 3: Дослідження стратегій кількості ітерацій N RANSAC (Частина C)
function tab_ransac_iterations()
  local rows = read_csv("c_ransac_iterations.csv")
  for _, r in ipairs(rows) do
    out({
      esc_tex(r[1]),
      f(r[4], 1),
      f(r[5], 1) .. " $\\pm$ " .. f(r[6], 1),
      r[8] .. "\\% (" .. r[9] .. "\\%)",
      f(r[10], 2) .. "~пк",
      f(r[11], 1) .. "~мс"
    })
  end
end

-- Таблиця 4: Звірка власного RANSAC з cv2.findHomography (Частина C)
function tab_ransac_comparison()
  local rows = read_csv("c_ransac_comparison.csv")
  for _, r in ipairs(rows) do
    out({
      esc_tex(r[1]),
      r[2],
      f(r[3], 2) .. "~пк",
      f(r[4], 2) .. "~пк",
      f(r[5], 1) .. "~мс"
    })
  end
end

-- Таблиця 5: Дослідження порога узгодженості tau (Частина C)
function tab_ransac_threshold()
  local rows = read_csv("c_ransac_threshold.csv")
  for _, r in ipairs(rows) do
    out({
      f(r[1], 1) .. "~пк",
      f(r[2], 1) .. " $\\pm$ " .. f(r[3], 1),
      f(r[4], 1) .. "\\%",
      f(r[5], 2) .. " $\\pm$ " .. f(r[6], 2) .. "~пк",
      r[7]
    })
  end
end

-- Таблиця 6: Об'єктивна оцінка якості панорами по парах (Частина D)
function tab_panorama_quality()
  local rows = read_csv("d_pair_quality.csv")
  for _, r in ipairs(rows) do
    out({
      "\\texttt{" .. esc_tex(r[1]) .. "}",
      r[2],
      r[3],
      f(r[4], 1) .. "\\%",
      f(r[5], 2) .. "~пк",
      f(r[6], 2) .. "~пк",
      f(r[7], 2) .. "~пк"
    })
  end
end

-- Таблиця 7: Дослідження критичних випадків відмови при дефіциті перекриття (Частина E)
function tab_overlap_failure()
  local rows = read_csv("e_overlap_failure.csv")
  for _, r in ipairs(rows) do
    local st = r[8]:gsub("кут кадру за лінією горизонту", "за горизонтом"):gsub("зміна площі ", "площа ")
    out({
      "\\texttt{" .. esc_tex(r[1]) .. "}",
      esc_tex(r[2]),
      r[3],
      r[4] .. " (" .. f(r[5], 1) .. "\\%)",
      esc_tex(r[6]),
      f(r[7], 2) .. "~пк",
      esc_tex(st)
    })
  end
end

-- Таблиця 8: Кількісний аналіз оптичного паралаксу на 3D-серії HDD (Частина E)
function tab_parallax_pairs()
  local rows = read_csv("e_parallax_pairs.csv")
  for _, r in ipairs(rows) do
    out({
      "\\texttt{" .. esc_tex(r[1]) .. "}",
      r[2],
      r[3] .. " (" .. f(r[4], 1) .. "\\%)",
      f(r[5], 2) .. "~пк",
      f(r[6], 1) .. "~пк"
    })
  end
end
